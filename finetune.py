import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from ui_layout_cnn import UILayoutCNN
import logging
from sklearn.metrics import classification_report

# === 로깅 설정 ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === 설정 ===
DATA_DIR = "data/train"
VAL_DIR = "data/validation"
BATCH_SIZE = 32
EPOCHS = 5           # 추가 학습은 에포크 수 짧게
LR = 1e-4            # 더 낮은 학습률
IMG_SIZE_H = 360
IMG_SIZE_W = 640
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['lol', 'tft']

# === 데이터 전처리 ===
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE_H, IMG_SIZE_W)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# === 모델 정의 및 로드 ===
model = UILayoutCNN(num_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load("best_model.pth"))  # 이전 모델 로드
model.to(DEVICE)
print("✅ 이전 best_model.pth 로드 완료")

# === 손실 함수 및 옵티마이저 설정 ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# === 이어 학습 루프 ===
best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    print(f"🔁 Epoch {epoch + 1}/{EPOCHS} 시작")

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    train_loss = running_loss / len(train_loader)

    print(f"✅ Epoch {epoch + 1} 완료 | Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

    # 검증
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True)
    val_acc = val_report['accuracy']  # type: ignore
    print(f"📊 Validation Accuracy: {val_acc:.4f}")

    # 최고 모델 저장
    if val_acc > best_acc:  # type: ignore
        best_acc = val_acc
        torch.save(model.state_dict(), "best_model_finetuned.pth")
        print(f"🏆 새로운 최고 검증 정확도: {best_acc:.4f} → 모델 저장 완료")

print("🎉 이어 학습 완료!")
