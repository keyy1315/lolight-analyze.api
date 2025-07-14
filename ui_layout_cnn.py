import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

"""
UILayoutCNN 모델 정의
- 입력 이미지를 4개의 영역으로 나누어 각 영역에서 feature map을 추출
- 추출된 feature map을 하나의 벡터로 변환
- 변환된 벡터를 분류기에 입력하여 분류 결과를 출력
- 모델 구조:
    - ResNet18 기반 CNN 인코더
    - 4개 영역에서 feature map 추출
    - 추출된 feature map을 하나의 벡터로 변환
    - 변환된 벡터를 분류기에 입력하여 분류 결과를 출력
"""
class UILayoutCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(UILayoutCNN, self).__init__()

        # 공통 CNN encoder: ResNet18 feature extractor (fc 제거)
        base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2])  # feature map만 추출

        # 각 crop 영역에서 feature map 뽑기 (4개 영역)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        # 최종 분류기 (512 * 4로 변경)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, image):
        # image: (B, 3, H, W)
        crops = self._crop_regions(image)  # (4, B, 3, h, w)
        features = []

        for crop in crops:
            feat = self.feature_extractor(crop)  # (B, 512, H', W')
            feat = self.avgpool(feat)            # (B, 512, 1, 1)
            feat = self.flatten(feat)            # (B, 512)
            features.append(feat)

        # Concatenate features from all regions
        combined = torch.cat(features, dim=1)     # (B, 512 * 4)
        out = self.classifier(combined)           # (B, num_classes)
        return out

    def _crop_regions(self, image):
        # Assumes input image is (B, 3, 360, 640)
        B, C, H, W = image.shape

        def crop_by_ratio(x1_ratio, y1_ratio, x2_ratio, y2_ratio):
            x1, x2 = int(W * x1_ratio), int(W * x2_ratio)
            y1, y2 = int(H * y1_ratio), int(H * y2_ratio)
            return image[:, :, y1:y2, x1:x2]

        # 우측하단: [240:360, 480:640] → (120x160)
        crop1 = crop_by_ratio(0.75, 0.66, 1.0, 1.0)

        # 상단 중앙: [0:120, 200:440] → (120x240)
        crop2 = crop_by_ratio(0.33, 0.0, 0.66, 0.33)

        # 하단 중앙: [240:360, 200:440] → (120x240)
        crop3 = crop_by_ratio(0.35, 0.66, 0.65, 1.0)

        # 좌측 중앙: [120:240, 0:213] (세로 3분할 중 가장 왼쪽)
        crop4 = crop_by_ratio(0.0, 0.33, 0.33, 0.66)

        return [crop1, crop2, crop3, crop4]

