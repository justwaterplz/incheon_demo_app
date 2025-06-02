import torch
import torch.nn as nn
import onnx
import onnx_tf
import tensorflow as tf
import numpy as np
import os

# Emergency Detection Model Architecture (same as before)
class EmergencyDetectionModel(nn.Module):
    def __init__(self, num_classes=2):
        super(EmergencyDetectionModel, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
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

def convert_pytorch_to_tflite(model_path="model.pt", output_path="app/src/main/assets/emergency_model.tflite"):
    """
    PyTorch 모델을 TensorFlow Lite로 변환
    단계: PyTorch → ONNX → TensorFlow → TFLite
    """
    
    print("=== PyTorch → TensorFlow Lite 변환 시작 ===")
    
    try:
        # Step 1: PyTorch 모델 준비
        print("1. PyTorch 모델 로딩...")
        model = EmergencyDetectionModel(num_classes=2)
        
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location='cpu')
                if isinstance(state_dict, dict):
                    model.load_state_dict(state_dict)
                else:
                    model = state_dict
                print("✓ 모델 파라미터 로드 완료")
            except Exception as e:
                print(f"⚠️ 파라미터 로드 실패: {e}")
                print("기본 모델 구조 사용")
        else:
            print(f"⚠️ 모델 파일 없음: {model_path}")
            print("기본 모델 구조 사용")
        
        model.eval()
        
        # Step 2: PyTorch → ONNX
        print("2. PyTorch → ONNX 변환...")
        input_shape = (1, 3, 224, 224)
        dummy_input = torch.randn(input_shape)
        
        onnx_path = "emergency_model.onnx"
        torch.onnx.export(
            model, 
            dummy_input, 
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"✓ ONNX 모델 생성: {onnx_path}")
        
        # Step 3: ONNX → TensorFlow
        print("3. ONNX → TensorFlow 변환...")
        onnx_model = onnx.load(onnx_path)
        tf_model = onnx_tf.backend.prepare(onnx_model)
        
        tf_model_path = "emergency_model_tf"
        tf_model.export_graph(tf_model_path)
        print(f"✓ TensorFlow 모델 생성: {tf_model_path}")
        
        # Step 4: TensorFlow → TensorFlow Lite
        print("4. TensorFlow → TensorFlow Lite 변환...")
        
        # 저장된 모델에서 TFLite 컨버터 생성
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
        
        # 최적화 설정
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]  # Float16 양자화
        
        # 변환 실행
        tflite_model = converter.convert()
        
        # assets 디렉토리 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # TFLite 모델 저장
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✓ TensorFlow Lite 모델 생성: {output_path}")
        
        # Step 5: 모델 테스트
        print("5. TensorFlow Lite 모델 테스트...")
        test_tflite_model(output_path)
        
        # 임시 파일 정리
        try:
            os.remove(onnx_path)
            import shutil
            shutil.rmtree(tf_model_path)
            print("✓ 임시 파일 정리 완료")
        except:
            pass
        
        print("\n🎉 PyTorch → TensorFlow Lite 변환 완료!")
        return True
        
    except Exception as e:
        print(f"❌ 변환 실패: {e}")
        print("\n필요한 패키지를 설치하세요:")
        print("pip install onnx onnx-tf tensorflow")
        return False

def test_tflite_model(tflite_path):
    """TensorFlow Lite 모델 테스트"""
    try:
        # TFLite 인터프리터 생성
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # 입력/출력 정보 가져오기
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"✓ 입력 형태: {input_details[0]['shape']}")
        print(f"✓ 출력 형태: {output_details[0]['shape']}")
        
        # 테스트 입력 생성
        input_shape = input_details[0]['shape']
        input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        
        # 추론 실행
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # 결과 가져오기
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(f"✓ 추론 성공 - 출력: {output_data.flatten()}")
        
        # 확률로 변환
        probabilities = tf.nn.softmax(output_data).numpy()
        print(f"✓ 확률: {probabilities.flatten()}")
        
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    convert_pytorch_to_tflite() 