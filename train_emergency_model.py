import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# 응급상황 감지 모델 아키텍처
class EmergencyDetectionModel(nn.Module):
    """
    응급상황 감지를 위한 CNN 모델
    입력: 224x224x3 이미지
    출력: [정상, 응급상황] 이진 분류
    """
    def __init__(self, num_classes=2):
        super(EmergencyDetectionModel, self).__init__()
        
        # 특징 추출 레이어
        self.features = nn.Sequential(
            # 첫 번째 블록
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 두 번째 블록
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 세 번째 블록
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 네 번째 블록
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 분류기
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 데이터셋 클래스
class EmergencyDataset(Dataset):
    """
    응급상황 감지를 위한 커스텀 데이터셋
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 이미지 로드
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 라벨
        label = self.labels[idx]
        
        # 전처리 적용
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_synthetic_emergency_data(num_samples=1000):
    """
    테스트용 합성 데이터 생성
    실제 환경에서는 실제 응급상황 영상 데이터를 사용해야 합니다.
    """
    print("테스트용 합성 데이터 생성 중...")
    
    data_dir = "emergency_data"
    os.makedirs(f"{data_dir}/normal", exist_ok=True)
    os.makedirs(f"{data_dir}/emergency", exist_ok=True)
    
    image_paths = []
    labels = []
    
    for i in tqdm(range(num_samples), desc="합성 데이터 생성"):
        # 정상 상황 (50%)
        if i < num_samples // 2:
            # 일반적인 활동을 시뮬레이션하는 이미지
            img = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
            # 부드러운 패턴 추가
            for _ in range(5):
                cv2.circle(img, 
                          (np.random.randint(0, 224), np.random.randint(0, 224)), 
                          np.random.randint(10, 30), 
                          (np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(100, 255)), 
                          -1)
            
            path = f"{data_dir}/normal/normal_{i}.jpg"
            cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            image_paths.append(path)
            labels.append(0)  # 정상
        
        # 응급 상황 (50%)
        else:
            # 응급상황을 시뮬레이션하는 이미지 (더 강렬한 색상과 패턴)
            img = np.random.randint(0, 100, (224, 224, 3), dtype=np.uint8)
            # 급격한 패턴 추가 (낙상, 충돌 등을 시뮬레이션)
            for _ in range(10):
                cv2.rectangle(img,
                             (np.random.randint(0, 200), np.random.randint(0, 200)),
                             (np.random.randint(50, 224), np.random.randint(50, 224)),
                             (np.random.randint(200, 255), np.random.randint(0, 50), np.random.randint(0, 50)),
                             -1)
            
            path = f"{data_dir}/emergency/emergency_{i}.jpg"
            cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            image_paths.append(path)
            labels.append(1)  # 응급상황
    
    return image_paths, labels

def train_model():
    """
    응급상황 감지 모델 훈련
    """
    print("=== 응급상황 감지 모델 훈련 시작 ===")
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")
    
    # 데이터 생성
    image_paths, labels = create_synthetic_emergency_data(2000)
    
    # 데이터 분할
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # 데이터 전처리
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 데이터셋 및 데이터로더 생성
    train_dataset = EmergencyDataset(train_paths, train_labels, train_transform)
    val_dataset = EmergencyDataset(val_paths, val_labels, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print(f"훈련 샘플: {len(train_dataset)}, 검증 샘플: {len(val_dataset)}")
    
    # 모델 생성
    model = EmergencyDetectionModel(num_classes=2).to(device)
    
    # 손실 함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # 훈련
    num_epochs = 30
    train_losses = []
    val_accuracies = []
    best_accuracy = 0.0
    
    print("\n훈련 시작...")
    for epoch in range(num_epochs):
        # 훈련 단계
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # 검증 단계
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        val_accuracies.append(accuracy)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        # 최고 성능 모델 저장
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_emergency_model.pt')
            print(f"새로운 최고 성능 모델 저장! (정확도: {accuracy:.2f}%)")
        
        scheduler.step()
    
    # 최종 모델 저장
    torch.save(model.state_dict(), 'final_emergency_model.pt')
    print(f"\n훈련 완료! 최고 정확도: {best_accuracy:.2f}%")
    
    # 손실 및 정확도 그래프
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()
    
    return model

def test_model():
    """
    훈련된 모델 테스트
    """
    print("\n=== 모델 테스트 ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 모델 로드
    model = EmergencyDetectionModel(num_classes=2).to(device)
    model.load_state_dict(torch.load('best_emergency_model.pt', map_location=device))
    model.eval()
    
    # 테스트 이미지 생성
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 정상 상황 테스트
    normal_img = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
    normal_tensor = test_transform(normal_img).unsqueeze(0).to(device)
    
    # 응급 상황 테스트
    emergency_img = np.random.randint(0, 100, (224, 224, 3), dtype=np.uint8)
    emergency_tensor = test_transform(emergency_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # 정상 상황 예측
        normal_output = model(normal_tensor)
        normal_prob = torch.softmax(normal_output, dim=1)
        normal_pred = torch.argmax(normal_prob, dim=1)
        
        # 응급 상황 예측
        emergency_output = model(emergency_tensor)
        emergency_prob = torch.softmax(emergency_output, dim=1)
        emergency_pred = torch.argmax(emergency_prob, dim=1)
        
        print(f"정상 상황 예측: {'정상' if normal_pred.item() == 0 else '응급'} (확률: {normal_prob[0][normal_pred].item():.3f})")
        print(f"응급 상황 예측: {'정상' if emergency_pred.item() == 0 else '응급'} (확률: {emergency_prob[0][emergency_pred].item():.3f})")

if __name__ == "__main__":
    # 모델 훈련
    trained_model = train_model()
    
    # 모델 테스트
    test_model()
    
    print("\n다음 단계:")
    print("1. convert_model.py를 실행하여 Android용 모델로 변환")
    print("2. convert_to_tflite.py를 실행하여 TensorFlow Lite 모델로 변환")
    print("3. Android 앱에서 테스트") 