# PyTorch 및 딥러닝 관련 임포트
import torch
from torch import nn, optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# 머신러닝 평가 및 유틸리티 임포트
from sklearn.metrics import classification_report
import logging
import time

# 커스텀 모델 임포트
from ui_layout_cnn import UILayoutCNN

"""
학습 설정
- 데이터 경로: data/train, data/validation
- 배치 크기: 32
- 학습 에포크 수: 10
- 학습률: 1e-3
- 이미지 크기: 360x640
- 디바이스: GPU/CPU 선택
- 클래스: lol, tft
- 데이터 증강: 좌우 반전, ±15도 회전, 색상 변화
- 정규화: ImageNet 정규화
- 손실 함수: CrossEntropyLoss
- 옵티마이저: Adam
- 최고 성능 모델 저장: best_model.pth
- 최종 모델 저잴: classify_model.pth
- 검증 성능 리포트: classification_report
"""

# === 로깅 설정 ===
# 학습 과정을 파일과 콘솔에 동시에 기록
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),  # 파일에 로그 저장
        logging.StreamHandler()               # 콘솔에 로그 출력
    ]
)

# === 학습 설정 ===
# 데이터 경로 설정
DATA_DIR = "data/train"           # 훈련 데이터 디렉토리
VAL_DIR = "data/validation"       # 검증 데이터 디렉토리

# 하이퍼파라미터 설정
BATCH_SIZE = 32                   # 배치 크기
EPOCHS = 10                       # 학습 에포크 수
LR = 1e-3                         # 학습률

# 이미지 크기 설정
IMG_SIZE_H = 360                  # 이미지 높이
IMG_SIZE_W = 640                  # 이미지 너비

# 디바이스 및 클래스 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU/CPU 선택
CLASS_NAMES = ['lol', 'tft']      # 분류할 클래스 이름

# 학습 시작 정보 출력
print(f"🚀 학습 시작")
print(f"📊 설정: BATCH_SIZE={BATCH_SIZE}, EPOCHS={EPOCHS}, LR={LR}")
print(f"💻 디바이스: {DEVICE}")
print(f"📝 클래스: {CLASS_NAMES}")

# === 데이터 전처리 ===
# 이미지 변환 파이프라인 정의 (데이터 증강 포함)
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE_H, IMG_SIZE_W)),  # 이미지 크기 조정
    transforms.RandomHorizontalFlip(p=0.5),       # 50% 확률로 좌우 반전
    transforms.RandomRotation(15),                # ±15도 회전
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 색상 변화
    transforms.ToTensor(),                        # PIL 이미지를 텐서로 변환
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet 정규화
])

# 데이터셋 로딩
print("📁 데이터셋 로딩 중...")
full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)    # 훈련 데이터셋
val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)      # 검증 데이터셋

print(f"📊 데이터 분할: Train={len(full_dataset)}, Val={len(val_dataset)}")

# 데이터 로더 생성
train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)   # 훈련 데이터 로더
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)     # 검증 데이터 로더

# === 모델 설정 ===
print("🤖 UILayoutCNN 모델 로딩 중...")
model = UILayoutCNN(num_classes=len(CLASS_NAMES))  # 분류 클래스 수에 맞는 모델 생성
model.to(DEVICE)                                   # 지정된 디바이스로 모델 이동

# 손실 함수 및 옵티마이저 설정
criterion = nn.CrossEntropyLoss()                  # 다중 클래스 분류를 위한 손실 함수
optimizer = optim.Adam(model.parameters(), lr=LR)  # Adam 옵티마이저

print(f"📈 모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

"""
학습 루프
- 배치별 학습
- 에포크별 학습 결과 출력
- 최고 성능 모델 저장 (Early Stopping)
- Early Stopping: 5에포크 동안 성능 개선이 없으면 학습 중단
- 검증 데이터로 모델 성능 평가
- 분류 성능 리포트 출력
"""

# === 학습 루프 ===
print("🎯 학습 시작!")
best_acc = 0.0  # 최고 정확도 기록용

for epoch in range(EPOCHS):
    epoch_start_time = time.time()  # 에포크 시작 시간 기록
    model.train()                    # 모델을 훈련 모드로 설정
    total_loss = 0.0                # 에포크 총 손실
    correct = 0                     # 정확히 분류된 샘플 수
    total = 0                       # 전체 샘플 수

    print(f"🔄 Epoch {epoch+1}/{EPOCHS} 시작")
    
    # 배치별 학습
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)  # 디바이스로 데이터 이동
        outputs = model(images)                                # 모델 예측
        loss = criterion(outputs, labels)                      # 손실 계산

        # 역전파 및 가중치 업데이트
        optimizer.zero_grad()  # 그래디언트 초기화
        loss.backward()        # 역전파
        optimizer.step()       # 가중치 업데이트

        # 통계 업데이트
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)  # 예측 클래스
        correct += (preds == labels).sum().item()  # 정확한 예측 수
        total += labels.size(0)  # 전체 샘플 수

        # 배치별 진행상황 로그 (5배치마다)
        if (batch_idx + 1) % 5 == 0:
            current_acc = correct / total
            print(f"   Batch {batch_idx+1}/{len(train_loader)}, "
                        f"Loss: {loss.item():.4f}, Acc: {current_acc:.4f}")

    # 에포크 결과 계산
    epoch_time = time.time() - epoch_start_time
    epoch_acc = correct / total
    epoch_loss = total_loss / len(train_loader)
    
    print(f"✅ Epoch {epoch+1}/{EPOCHS} 완료")
    print(f"   📊 Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
    print(f"   ⏱️  소요시간: {epoch_time:.2f}초")

    # 최고 성능 모델 저장 (Early Stopping)
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        patience = 0  # 패턴스 카운터 리셋
        torch.save(model.state_dict(), "best_model.pth")  # 최고 성능 모델 저장
        print(f"🏆 새로운 최고 정확도: {best_acc:.4f}")
    else:
        patience += 1  # 성능 개선이 없으면 패턴스 증가
    
    # Early Stopping: 5에포크 동안 성능 개선이 없으면 학습 중단
    if patience >= 5:
        print("🔄 Early Stopping - 성능 개선이 없어 학습을 중단합니다")
        break

# === 검증 ===
print("🔍 검증 시작...")
model.eval()  # 모델을 평가 모드로 설정
all_preds = []    # 모든 예측 결과 저장
all_labels = []   # 모든 실제 라벨 저장

# 검증 데이터로 모델 성능 평가
with torch.no_grad():  # 그래디언트 계산 비활성화
    for images, labels in val_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)  # 디바이스로 데이터 이동
        outputs = model(images)                                # 모델 예측
        preds = torch.argmax(outputs, dim=1)                  # 예측 클래스
        all_preds.extend(preds.cpu().numpy())                 # 예측 결과 저장
        all_labels.extend(labels.cpu().numpy())                # 실제 라벨 저장

# 분류 성능 리포트 출력
print("📋 Classification Report (Validation):")
print("\n🔍 Classification Report (Validation):")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

print("🎉 학습 완료!")

# 최종 모델 저장
torch.save(model.state_dict(), "classify_model.pth")