import os
import cv2
import random
import string

def extract_and_save_frames(video_path, output_folder, frame_count=5):
    os.makedirs(output_folder, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    selected_frames = sorted(random.sample(range(total_frames), min(frame_count, total_frames)))

    random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
    
    count = 0
    images = []
    for frame_num in selected_frames:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        success, image = vidcap.read()
        if success:
            image = cv2.resize(image, (1280, 720))
            images.append(image)
            filename = os.path.join(output_folder, f"{random_str}_{count}.jpg")
            cv2.imwrite(filename, image)
            count += 1

    vidcap.release()
    print(f"✅ Extracted {count} frames to {output_folder}")
    return images

def extract_frames_every_second(video_path, output_folder, max_seconds=None):
    # 영상에서 3초마다 프레임 추출
    os.makedirs(output_folder, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    
    # 비디오 정보
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / fps
    
    print(f"📹 비디오 정보: {fps:.2f} FPS, 총 {total_duration:.2f}초")
    
    # 3초마다 프레임 선택 (FPS * 3만큼 건너뛰기)
    frames_per_3_seconds = int(fps * 3)
    selected_frames = []
    
    if max_seconds:
        max_frames = min(max_seconds * frames_per_3_seconds, total_frames)
        for i in range(0, max_frames, frames_per_3_seconds):
            selected_frames.append(i)
    else:
        for i in range(0, total_frames, frames_per_3_seconds):
            selected_frames.append(i)
    
    random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
    
    count = 0
    for frame_num in selected_frames:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        success, image = vidcap.read()
        if success:
            image = cv2.resize(image, (1920, 1080))
            
            # 시간 정보를 파일명에 포함 (3초 단위)
            second = frame_num // frames_per_3_seconds * 3
            filename = os.path.join(output_folder, f"{random_str}_{second:03d}s_{count}.jpg")
            cv2.imwrite(filename, image)
            count += 1

    vidcap.release()
    print(f"✅ 3초마다 {count}개 프레임을 {output_folder}에 추출했습니다")
    return count
