#!/usr/bin/env python3
"""
5D 텐서를 지원하는 PyTorch Mobile 모델 생성 스크립트
원본 baseline_3d_resnets.py 기반으로 모바일용 .ptl 파일 생성
"""

import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile
import sys
import os

# baseline_3d_resnets.py에서 필요한 클래스들 import
# (실제로는 baseline_3d_resnets.py 파일이 같은 디렉토리에 있어야 함)
from baseline_3d_resnets import resnet50, resnet18, resnet34

def create_mobile_5d_model(
    model_type='resnet50',
    num_classes=8,
    input_size=112, 
    input_len=16,
    pretrained=False,
    output_path='8cls_5d.ptl'
):
    """
    5D 텐서를 지원하는 모바일용 모델 생성
    
    Args:
        model_type: 'resnet18', 'resnet34', 'resnet50' 중 선택
        num_classes: 출력 클래스 수 (8개 클래스)
        input_size: 입력 이미지 크기 (112x112)
        input_len: 입력 프레임 수 (16)
        pretrained: ImageNet 사전훈련 모델 사용 여부
        output_path: 출력 .ptl 파일 경로
    """
    
    print(f"🚀 5D 모바일 모델 생성 시작...")
    print(f"   - 모델: {model_type}")
    print(f"   - 클래스 수: {num_classes}")
    print(f"   - 입력 크기: {input_size}x{input_size}x{input_len}")
    
    # 1. 모델 생성
    if model_type == 'resnet18':
        model = resnet18(pretrained=pretrained, num_classes=num_classes, 
                        input_size=input_size, input_len=input_len)
    elif model_type == 'resnet34':
        model = resnet34(pretrained=pretrained, num_classes=num_classes,
                        input_size=input_size, input_len=input_len)
    elif model_type == 'resnet50':
        model = resnet50(pretrained=pretrained, num_classes=num_classes,
                        input_size=input_size, input_len=input_len)
    else:
        raise ValueError(f"지원하지 않는 모델: {model_type}")
    
    # 2. 훈련된 가중치 로드 (있는 경우)
    weight_path = f"trained_{model_type}_{num_classes}cls.pth"
    if os.path.exists(weight_path):
        print(f"📂 훈련된 가중치 로드: {weight_path}")
        model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    else:
        print(f"⚠️ 훈련된 가중치를 찾을 수 없음: {weight_path}")
        print(f"   랜덤 초기화된 모델을 사용합니다.")
    
    # 3. 평가 모드로 설정
    model.eval()
    
    # 4. 5D 더미 입력 생성 (B, C, T, H, W)
    dummy_input = torch.randn(1, 3, input_len, input_size, input_size)
    print(f"📊 테스트 입력 형태: {dummy_input.shape}")
    
    # 5. 모델 추론 테스트
    with torch.no_grad():
        try:
            output = model(dummy_input)
            print(f"✅ 모델 추론 성공! 출력 형태: {output.shape}")
        except Exception as e:
            print(f"❌ 모델 추론 실패: {e}")
            return False
    
    # 6. TorchScript로 변환 (trace 방식)
    print(f"🔄 TorchScript 변환 중...")
    try:
        traced_model = torch.jit.trace(model, dummy_input)
        print(f"✅ TorchScript 변환 성공!")
    except Exception as e:
        print(f"❌ TorchScript 변환 실패: {e}")
        return False
    
    # 7. 모바일 최적화
    print(f"📱 모바일 최적화 중...")
    try:
        optimized_model = optimize_for_mobile(traced_model)
        print(f"✅ 모바일 최적화 성공!")
    except Exception as e:
        print(f"❌ 모바일 최적화 실패: {e}")
        return False
    
    # 8. .ptl 파일 저장
    print(f"💾 모델 저장 중: {output_path}")
    try:
        optimized_model._save_for_lite_interpreter(output_path)
        
        # 파일 크기 확인
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"✅ 모델 저장 성공!")
        print(f"📄 파일 크기: {file_size:.1f}MB")
        
    except Exception as e:
        print(f"❌ 모델 저장 실패: {e}")
        return False
    
    # 9. 저장된 모델 검증
    print(f"🔍 저장된 모델 검증 중...")
    try:
        loaded_model = torch.jit.load(output_path)
        loaded_model.eval()
        
        with torch.no_grad():
            verification_output = loaded_model(dummy_input)
            print(f"✅ 모델 검증 성공! 출력 형태: {verification_output.shape}")
            
        return True
        
    except Exception as e:
        print(f"❌ 모델 검증 실패: {e}")
        return False

def main():
    """메인 실행 함수"""
    
    # 기본 설정
    configs = [
        {
            'model_type': 'resnet18',
            'num_classes': 8,
            'input_size': 112,
            'input_len': 16,
            'pretrained': True,
            'output_path': '8cls_resnet18_5d.ptl'
        },
        {
            'model_type': 'resnet50', 
            'num_classes': 8,
            'input_size': 112,
            'input_len': 16,
            'pretrained': True,
            'output_path': '8cls_resnet50_5d.ptl'
        }
    ]
    
    print("🎯 5D PyTorch Mobile 모델 생성기")
    print("=" * 50)
    
    success_count = 0
    for i, config in enumerate(configs, 1):
        print(f"\n📋 설정 {i}/{len(configs)}")
        print("-" * 30)
        
        success = create_mobile_5d_model(**config)
        if success:
            success_count += 1
            print(f"🎉 {config['output_path']} 생성 완료!")
        else:
            print(f"💥 {config['output_path']} 생성 실패!")
    
    print(f"\n📊 결과: {success_count}/{len(configs)} 모델 생성 성공")
    
    if success_count > 0:
        print("\n📱 Android 프로젝트 적용 방법:")
        print("1. 생성된 .ptl 파일을 app/src/main/assets/에 복사")
        print("2. ActionClassifier.kt에서 MODEL_NAME 수정")
        print("3. 5D 텐서 지원 코드로 업데이트")

if __name__ == '__main__':
    main() 