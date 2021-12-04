<h1 align="center">
BoostCamp AI Tech - Model Optimization 
</h1>

# 1. 프로젝트 개요
여러 산업에서 인공지능을 이용해 문제를 해결하고 있다. 예를들어, 수퍼큐브는 수퍼빈에서 만든 인공지능 분리수거 기계로 사람이 기계에 캔과 페트병을 넣으면 내부에서 인공지능을 통해 재활용이 가능한 쓰레기인지를 판단해 보관해주는 서비스이다. 이 서비스에서 가장 중요한 것은 들어온 쓰레기를 잘 분류해내는 것이다.  하지만,  실제 서비스로 활용하기 위해서는 해당 인공지능 모델이 로봇 내부 시스템에 탑재되어 즉각적으로 쓰레기를 분류할 수 있어야만 한다는 점이다. 일반적인 연구환경보다 로봇 내부 시스템은 상대적으로 하드웨어가 떨어질 수 밖에 없으며 단순히 예측력만 높은 무거운 모델은 활용하기 어려울 것이다.
이번 프로젝트는 소형 분리수거 로봇에 탑재되는 쓰레기 분류기의 계산량을 최대한 줄이되, 일정 수준 이상의 성능을 내는 것을 목표로 한다.

## 1.1 프로젝트 주제
구동하려는 타겟 디바이스 위에서 작동하는 모델의 효율을 개선하는 task이며, 이번 프로젝트는 V100을 기준으로 속도의 정확도를 개선하는 것을 목표로 하고 있다.

## 1.2 학습 데이터 
COCO format의 TACO Dataset에서 쓰레기 부분만을 crop한 데이터를 사용했다. 그리고 train data는 6개의 카테고리로 분류되어 있고 총 20,851장의 .jpg format 이미지로 구성되어 있다.

## 1.3 평가방법
학습데이터의 평가방법으로는 입력된 쓰레기 이미지에 대한 분류 성능 (f1 score)과 추론 속도 (submit time)으로 측정한다.
![](https://i.imgur.com/G6Qbmal.png)

Score (f1-score) : 분류 성능 지표로, 기준이 되는 모델의 f1 score에서 제출한 모델의 f1 score의 차이를 구한 뒤, 상수를 곱하고 sigmoid 함수를 적용한 값 (낮을수록 좋음)
Score (submit time) : 기준이 되는 모델의 추론하는 시간으로 제출한 모델의 추론하는 시간을 나눈 값 (낮을수록 좋음)

## 1.4. 프로젝트 수행과정
Baseline code 분석 - Optuna Model search - Optuna Parameter search - Optuna 

#

# 2. 프로젝트 팀 구성 및 역할
## 2.1. 팀 소개 : 6조 자연어학당
- 나요한_T2073 : Baseline code 추가 구현, Optuna Model, Hyperparameter search 실험, 
		EfficientNet b0 fine-tuning + hyperparameter tuning
- 백재형_T2102 :  Optuna Model search 실험
- 송민재_T2116 :  Unstructured Pruning 실험
- 이호영_T2177 : Optuna Hyperparameter search 실험 진행 , EfficientNet b0 fine-tuning
- 정찬미_T2207 : Optuna Hyperparameter search 실험
- 한진_T2237  :  squeezenet, mobilenet v3 small 모델 실험
- 홍석진_T2243 : MobileNetV2 기반 모델 커스터마이징 및 hyperparameter, augmentation 변경 실험



# 3. 프로젝트 수행 절차 및 방법
11.22(월) 최적화 소개
11.23(화) 데이터셋 분석, AutoML이론
11.24(수) AutoML실습
11.25(목) Baseline코드 이해 및 학습
11.26(금) Optuna 튜닝
11.29(월) Optuna Visualization
11.30(화) Optuna Parallel + PostgreSQL
12.01(수) Albumentation try
12.02(목) Retraining
12.03(금) Wrap-up Report 

# 4. 프로젝트 수행 결과

## 4.1. 모델링
![](https://i.imgur.com/DHmdUbj.png)

- Custom Conv Model : 44k parameter, 단순구조 
- mobilenet v2 : 221 layers, 3.5M parameters인 mobilenet v2모델을 반복 횟수를 줄여 101 layers, 2.4M parameters로 개선
- optuna : 52k paramters, 경량모델 위주의 탐색
- efficientnet_b0 : 4M parameters, Pre-trained Model을 활용해 최고 성능확인


## 4.2. Pruning
실험에 사용했던 모델들의 파라미터 갯수는 2M ~ 4M으로 높았기 때문에 파라미터수를 줄이고자  pruning을 torch.nn.utils.prune 모듈을 이용해 구현했다. 각 Conv Layer에 1l unstructured을 사용해 20~30%의 파라미터수를 줄이고자 했다. 파라미터를 없애는게 아니라 0으로 바꾸는 unstructured  pruning이었으나 Sparse Matrix를 계산하는데 특화된 하드웨어가 아니여서 연산속도향상을 이뤄내지는 못하였다.

## 4.3. Optuna 활용
AutoML을 활용해 모델을 찾아보기를 시도했다. 최대 7개의 레이어를 구성할 수 있고, F1 Score, Parameter size, mean time, 각각을 [최대, 최소, 최소]화 하는 방향으로 탐색적 학습을 진행하였다. 100회 정도의 시도결과 97번째 시도에서 준수한 성능과 inference time을 가지는 모델을 찾아낼 수 있었다.
![](https://i.imgur.com/kNYBnA0.png)


## 4.4. Augmentation 변경을 통한 재학습  
![](https://i.imgur.com/ZJ27inI.png)
학습 epoch이 증가해도 0.65에서 멈춰있던 모델에서 Augmentation 방식 변경
->  randaugmentation으로 재학습한 결과 약 3%p의 성능 향상


## 4.5. 수행결과
- mobilenet v2 : score 1.3619
- optuna : score 1.4586
![](https://i.imgur.com/4dB7Soo.png)



# Environment
## 1. Docker
```bash
docker run -it --gpus all --ipc=host -v ${path_to_code}:/opt/ml/code -v ${path_to_dataset}:/opt/ml/data placidus36/pstage4_lightweight:v0.4 /bin/bash
```
## 2. Install dependencies
```
pip install -r requirements.txt
```

# Run
## 1. train
python train.py --model_config ${path_to_model_config} --data_config ${path_to_data_config}

## 2. inference(submission.csv)
python inference.py --model_config configs/model/mobilenetv3.yaml --weight exp/2021-05-13_16-41-57/best.pt --img_root /opt/ml/data/test --data_config configs/data/taco.yaml3

# Reference
Our basic structure is based on [Kindle](https://github.com/JeiKeiLim/kindle)(by [JeiKeiLim](https://github.com/JeiKeiLim))

