from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uvicorn
import tempfile
import shutil
import os
from utils import extract_and_save_frames, extract_frames_every_second
import tensorflow as tf
import numpy as np
from keras.models import load_model
import cv2

app = FastAPI(
    title="Video Classifier API",
    description="API for video classification",
    version="1.0.0"
)

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
    server test api
    curl -X GET http://localhost:8082/

    """
    return {"message": "Hello, World!"}

@app.post("/result")
async def result(file: UploadFile = File(...)):
    """
    video classification api
    curl -X POST http://localhost:8082/result -F "file=@./data/train/lol/000000000000.mp4"
    """
    model = load_model('./model/video_classifier.keras')  # type: ignore
    labels = ['lol', 'tft', 'unknown']
    
    allowed_extensions = {'.webm', '.mp4', '.avi', '.mov', '.mkv'}

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided"
        )
    
    file_extension = '.' + file.filename.split('.')[-1].lower() if '.' in file.filename else ''
    
    if file_extension not in allowed_extensions:
        raise HTTPException( 
            status_code=400,
            detail=f"Unsupported file format. Allowed formats: {', '.join(allowed_extensions)}"
        )
    
    max_size = 500 * 1024 * 1024
    if file.size and file.size > max_size:
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum size is 100MB."
        )

    try:
        temp_file = None
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
            shutil.copyfileobj(file.file, temp_file)
            temp_file.close()
            tmp_path = temp_file.name
         
            images = extract_and_save_frames(tmp_path, "data/temp")

            if len(images) == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Failed to extract frames from the video."
                )


            all_preds = []
            for image in images:
                image = cv2.resize(image, (640, 360))
                image = image.astype(np.float32) / 255.0
                image = image.reshape(1, 360, 640, 3)

                pred = model.predict(image)  # type: ignore
                all_preds.append(pred[0])
            
            avg_pred = np.mean(all_preds, axis=0)
            label_index = np.argmax(avg_pred)
            final_label = labels[label_index]

            return {
                "label": final_label,
                "score": float(avg_pred[label_index])
            }
        finally:
            if temp_file and hasattr(temp_file, 'name'):
                try:
                    if os.path.exists(temp_file.name):
                        os.unlink(temp_file.name)
                except Exception as cleanup_error:
                    print(f"Warning: Could not delete temporary file {temp_file.name}: {cleanup_error}")
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@app.post("/data-set")
async def input_video(files: List[UploadFile] = File(...), dir_name: str = Form(...)):
    allowed_extensions = {'.webm', '.mp4', '.avi', '.mov', '.mkv'}
    
    if not files:
        raise HTTPException(
            status_code=400,
            detail="No files provided"
        )
    
    # 파일 검증
    for file in files:
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="No filename provided"
            )
        
        file_extension = '.' + file.filename.split('.')[-1].lower() if '.' in file.filename else ''
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {file.filename}. Allowed formats: {', '.join(allowed_extensions)}"
            )
        
        max_size = 500 * 1024 * 1024
        if file.size and file.size > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {file.filename}. Maximum size is 500MB."
            )
    
    total_frame_count = 0
    processed_files = []
    
    try:
        for file in files:
            temp_file = None
            try:
                if not file.filename:
                    continue
                    
                file_extension = '.' + file.filename.split('.')[-1].lower() if '.' in file.filename else '.mp4'
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
                shutil.copyfileobj(file.file, temp_file)
                temp_file.close()
                tmp_path = temp_file.name

                frame_count = extract_frames_every_second(tmp_path, dir_name)

                if frame_count == 0:
                    print(f"Warning: Failed to extract frames from {file.filename}")
                else:
                    total_frame_count += frame_count
                    processed_files.append(file.filename)
                    print(f"✅ {file.filename}: {frame_count} frames extracted")
                
            finally:
                if temp_file and hasattr(temp_file, 'name'):
                    try:
                        if os.path.exists(temp_file.name):
                            os.unlink(temp_file.name)
                    except Exception as cleanup_error:
                        print(f"Warning: Could not delete temporary file {temp_file.name}: {cleanup_error}")
        
        if total_frame_count == 0:
            raise HTTPException(
                status_code=400,
                detail="Failed to extract frames from any video."
            )
        
        return {
            "total_frame_count": total_frame_count,
            "processed_files": processed_files,
            "total_files": len(files),
            "successful_files": len(processed_files)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8082)