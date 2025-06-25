from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import shutil
import os
from utils import extract_frames_every_second
import tensorflow as tf

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
    return {"message": "Hello, World!"}

@app.post("/result")
async def result(file: UploadFile = File(...)):
    model = tf.keras.models.load_model('./model/video_classifier.keras')

    class_names = ['lol', 'tft', 'unknown']

    # TODO: 영상에서 5-10 프레임 추출 후 모델에 넣어서 결과 반환
    return {"message": "Hello, World!"}

@app.post("/data-set")
async def input_video(file: UploadFile = File(...)):
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

            # frame_count = extract_and_save_frames(tmp_path, "data/train/lol")
            frame_count = extract_frames_every_second(tmp_path, "data/train/tft")

            if frame_count == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Failed to extract frames from the video."
                )
            
            
        finally:
            if temp_file and hasattr(temp_file, 'name'):
                try:
                    if os.path.exists(temp_file.name):
                        os.unlink(temp_file.name)
                except Exception as cleanup_error:
                    print(f"Warning: Could not delete temporary file {temp_file.name}: {cleanup_error}")
        
        return {
            "frame_count": frame_count
        }

    except Exception as e:
        if 'temp_file' in locals() and temp_file and hasattr(temp_file, 'name'):
            try:
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
            except Exception as cleanup_error:
                print(f"Warning: Could not delete temporary file {temp_file.name}: {cleanup_error}")

        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8082)