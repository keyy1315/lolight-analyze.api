# FastAPI 및 웹 프레임워크 임포트
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uvicorn

# 파일 처리 및 시스템 임포트
import tempfile
import shutil
import os

# 비디오 처리를 위한 커스텀 유틸리티 함수
from utils import extract_and_save_frames, extract_frames_every_second

# 데이터 처리 및 머신러닝 임포트
import numpy as np
from ui_layout_cnn import UILayoutCNN
import torch
from torchvision import transforms
from PIL import Image

# PyTorch 디바이스 설정 (GPU 사용 가능시 GPU, 아니면 CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FastAPI 애플리케이션 초기화
app = FastAPI(
    title="Video Classifier API",
    description="API for video classification",
    version="1.0.0"
)

# 크로스 오리진 요청을 허용하는 CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """
    서버 테스트를 위한 헬스 체크 엔드포인트
    curl -X GET http://localhost:8082/
    """
    return {"message": "Hello, World!"}

@app.post("/result")
async def result(file: UploadFile = File(...)):
    """
    비디오 분류 API 엔드포인트
    비디오 파일을 받아서 분류 결과를 반환 (lol/tft)
    curl -X POST http://localhost:8082/result -F "file=@./data/train/lol/000000000000.mp4"
    """
    # 훈련된 모델 로드 및 설정
    model = UILayoutCNN(num_classes=2)
    model.load_state_dict(torch.load('./best_model.pth'))
    model.to(DEVICE)
    model.eval()
    
    # 분류를 위한 클래스 라벨 정의
    labels = ['lol', 'tft']
    
    # 허용된 비디오 파일 확장자 정의
    allowed_extensions = {'.webm', '.mp4', '.avi', '.mov', '.mkv'}

    # 파일 업로드 검증
    if not file.filename:
        raise HTTPException(
            status_code=400, 
            detail="No filename provided"
        )
    
    # 파일 확장자 추출 및 검증
    file_extension = '.' + file.filename.split('.')[-1].lower() if '.' in file.filename else ''
    
    if file_extension not in allowed_extensions:
        raise HTTPException( 
            status_code=400,
            detail=f"Unsupported file format. Allowed formats: {', '.join(allowed_extensions)}"
        )
    
    # 파일 크기 제한 확인 (500MB)
    max_size = 500 * 1024 * 1024
    if file.size and file.size > max_size:
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum size is 500MB."
        )

    try:
        temp_file = None
        try:
            # 업로드된 비디오를 저장할 임시 파일 생성
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
            shutil.copyfileobj(file.file, temp_file)
            temp_file.close()
            tmp_path = temp_file.name

            # 모델을 사용하여 비디오에서 프레임 추출
            image_paths = extract_and_save_frames(tmp_path, "data/temp", model)

            # 프레임이 성공적으로 추출되었는지 검증
            if len(image_paths) == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Failed to extract frames from the video."
                )


            # 분류를 위해 추출된 각 프레임 처리
            all_preds = []
            for image_path in image_paths:
                # 이미지 전처리 파이프라인 정의
                transform = transforms.Compose([
                    transforms.Resize((360, 640)),  # 모델 입력 차원에 맞게 리사이즈
                    transforms.ToTensor(),           # PIL 이미지를 텐서로 변환
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 정규화
                ])

                # 이미지 로드 및 전처리
                image = Image.open(image_path).convert('RGB')
                tensor : torch.Tensor = transform(image) # type: ignore
                input_tensor = tensor.unsqueeze(0)  # 배치 차원 추가

                # 모델에서 추론 실행
                with torch.no_grad():
                    output = model(input_tensor)
                    pred = output.softmax(dim=1)  # 로짓을 확률로 변환
                    all_preds.append(pred.cpu().numpy())

            # 모든 프레임의 예측 결과 집계
            all_preds = np.array(all_preds)
            avg_pred = np.mean(all_preds, axis=0)  # 모든 프레임의 예측 평균
            avg_pred = avg_pred.squeeze()  # 단일 차원 축 제거

            # 최종 분류 결과 결정
            label_index = np.argmax(avg_pred)  # 최고 확률의 인덱스 가져오기
            confidence_score = float(avg_pred[label_index])  # 신뢰도 점수
            
            # 신뢰도 점수가 0.6 이하면 unknown으로 분류
            if confidence_score < 0.6:
                final_label = "unknown"
            else:
                final_label = labels[label_index]  # 인덱스를 라벨로 매핑

            print(f"🔎 모든 프레임의 shape : {all_preds.shape}")
            print(f"🔎 모든 프레임의 평균 shape : {avg_pred.shape}")

            # 분류 결과 반환
            return {
                "label": final_label,
                "score": confidence_score,  # 신뢰도 점수
                "lol_score": avg_pred[0],
                "tft_score": avg_pred[1]
            }
        finally:
            # 임시 파일 정리
            if temp_file and hasattr(temp_file, 'name'):
                try:
                    if os.path.exists(temp_file.name):
                        os.unlink(temp_file.name)
                except Exception as cleanup_error:
                    print(f"Warning: Could not delete temporary file {temp_file.name}: {cleanup_error}")
        
    except Exception as e:
        # 처리 중 발생한 오류 처리
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@app.post("/data-set")
async def input_video(files: List[UploadFile] = File(...), dir_name: str = Form(...)):
    """
    데이터셋 준비 엔드포인트
    여러 비디오 파일을 받아서 훈련 데이터셋을 위한 프레임을 추출
    """
    # 허용된 비디오 파일 확장자 정의
    allowed_extensions = {'.webm', '.mp4', '.avi', '.mov', '.mkv'}
    
    # 파일이 제공되었는지 검증
    if not files:
        raise HTTPException(
            status_code=400,
            detail="No files provided"
        )
    
    # 업로드된 각 파일 검증
    for file in files:
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="No filename provided"
            )
        
        # 파일 확장자 확인
        file_extension = '.' + file.filename.split('.')[-1].lower() if '.' in file.filename else ''
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {file.filename}. Allowed formats: {', '.join(allowed_extensions)}"
            )
        
        # 파일 크기 제한 확인
        max_size = 500 * 1024 * 1024
        if file.size and file.size > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {file.filename}. Maximum size is 500MB."
            )
    
    # 추적 변수 초기화
    total_frame_count = 0
    processed_files = []
    
    try:
        # 업로드된 각 비디오 파일 처리
        for file in files:
            temp_file = None
            try:
                if not file.filename:
                    continue
                    
                # 비디오 처리를 위한 임시 파일 생성
                file_extension = '.' + file.filename.split('.')[-1].lower() if '.' in file.filename else '.mp4'
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
                shutil.copyfileobj(file.file, temp_file)
                temp_file.close()
                tmp_path = temp_file.name

                # 비디오에서 프레임 추출 (초당 1프레임)
                frame_count = extract_frames_every_second(tmp_path, dir_name)

                # 처리 결과 추적
                if frame_count == 0:
                    print(f"Warning: Failed to extract frames from {file.filename}")
                else:
                    total_frame_count += frame_count
                    processed_files.append(file.filename)
                    print(f"✅ {file.filename}: {frame_count} frames extracted")
                
            finally:
                # 임시 파일 정리
                if temp_file and hasattr(temp_file, 'name'):
                    try:
                        if os.path.exists(temp_file.name):
                            os.unlink(temp_file.name)
                    except Exception as cleanup_error:
                        print(f"Warning: Could not delete temporary file {temp_file.name}: {cleanup_error}")
        
        # 최소한 일부 프레임이 추출되었는지 검증
        if total_frame_count == 0:
            raise HTTPException(
                status_code=400,
                detail="Failed to extract frames from any video."
            )
        
        # 처리 요약 반환
        return {
            "total_frame_count": total_frame_count,
            "processed_files": processed_files,
            "total_files": len(files),
            "successful_files": len(processed_files)
        }

    except Exception as e:
        # 처리 중 발생한 오류 처리
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

if __name__ == "__main__":
    # FastAPI 서버 시작
    uvicorn.run(app, host="0.0.0.0", port=8082)