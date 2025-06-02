import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile
import os

# Emergency Detection Model Architecture
# 이 클래스는 원본 GitHub 프로젝트의 모델 구조를 재현해야 합니다
class EmergencyDetectionModel(nn.Module):
    """
    Android용 응급상황 탐지 모델
    원본 GitHub 프로젝트 (incheon_publicdata)의 모델 구조를 기반으로 합니다.
    """
    def __init__(self, num_classes=2):
        super(EmergencyDetectionModel, self).__init__()
        
        # 기본적인 CNN 구조 (실제 모델 구조에 맞게 수정 필요)
        self.features = nn.Sequential(
            # 첫 번째 컨볼루션 블록
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 두 번째 컨볼루션 블록
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 세 번째 컨볼루션 블록
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 분류기
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def convert_model_to_mobile(model_path, output_path):
    """
    PyTorch 모델을 Android용 TorchScript Lite 형식으로 변환
    
    Args:
        model_path (str): 원본 .pt 파일 경로
        output_path (str): 변환된 .ptl 파일이 저장될 경로
    """
    try:
        print(f"모델 로딩 중: {model_path}")
        
        # 모델 인스턴스 생성
        model = EmergencyDetectionModel(num_classes=2)
        
        # 저장된 파라미터 로드
        if os.path.exists(model_path):
            # state_dict 로드 (파라미터만 있는 경우)
            try:
                state_dict = torch.load(model_path, map_location='cpu')
                if isinstance(state_dict, dict):
                    # state_dict만 있는 경우
                    model.load_state_dict(state_dict)
                else:
                    # 전체 모델이 저장된 경우
                    model = state_dict
                print("✓ 모델 파라미터 로드 완료")
            except Exception as e:
                print(f"⚠️ 파라미터 로드 실패: {e}")
                print("기본 모델 구조를 사용합니다 (테스트 모드)")
        else:
            print(f"⚠️ 모델 파일을 찾을 수 없습니다: {model_path}")
            print("기본 모델 구조를 사용합니다 (테스트 모드)")
        
        # 모델을 평가 모드로 설정
        model.eval()
        
        # 예제 입력 텐서 생성 (배치 크기 1, 3채널, 224x224)
        example_input = torch.randn(1, 3, 224, 224)
        
        print("TorchScript 변환 중...")
        
        # TorchScript로 변환 (trace 방식 사용)
        traced_model = torch.jit.trace(model, example_input)
        
        print("모바일 최적화 중...")
        
        # 모바일용 최적화
        optimized_model = optimize_for_mobile(traced_model)
        
        # Lite Interpreter용으로 저장
        optimized_model._save_for_lite_interpreter(output_path)
        
        print(f"✓ 변환 완료: {output_path}")
        
        # 변환된 모델 테스트
        print("변환된 모델 테스트 중...")
        test_lite_model(output_path)
        
        return True
        
    except Exception as e:
        print(f"❌ 변환 실패: {e}")
        return False

def test_lite_model(lite_model_path):
    """
    변환된 Lite 모델이 정상적으로 작동하는지 테스트
    """
    try:
        # Lite 모델 로드
        lite_model = torch.jit.load(lite_model_path)
        
        # 테스트 입력
        test_input = torch.randn(1, 3, 224, 224)
        
        # 추론 실행
        with torch.no_grad():
            output = lite_model(test_input)
        
        print(f"✓ 테스트 성공 - 출력 형태: {output.shape}")
        print(f"  출력 값 (로짓): {output.numpy().flatten()}")
        
        # 확률로 변환
        probabilities = torch.softmax(output, dim=1)
        print(f"  확률: {probabilities.numpy().flatten()}")
        
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False

def main():
    """메인 함수"""
    # 파일 경로 설정
    model_path = "model.pt"  # 원본 모델 파일
    output_path = "app/src/main/assets/emergency_model.ptl"  # 변환된 모델이 저장될 경로
    
    print("=== PyTorch 모델을 Android용 TorchScript Lite로 변환 ===")
    print(f"입력 모델: {model_path}")
    print(f"출력 경로: {output_path}")
    
    # assets 디렉토리 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 변환 실행
    success = convert_model_to_mobile(model_path, output_path)
    
    if success:
        print("\n🎉 모델 변환이 성공적으로 완료되었습니다!")
        print(f"Android 앱에서 사용할 수 있는 모델: {output_path}")
        print("\n다음 단계:")
        print("1. Android 프로젝트를 빌드하세요")
        print("2. 실제 디바이스에서 테스트하세요")
    else:
        print("\n❌ 모델 변환에 실패했습니다.")
        print("원본 GitHub 프로젝트에서 모델 구조를 확인하고 EmergencyDetectionModel 클래스를 수정하세요.")

if __name__ == "__main__":
    main() 