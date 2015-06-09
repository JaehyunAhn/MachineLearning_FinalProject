## 기계학습 기말 프로젝트

- 지은이: 안재현(jaehyunahn@sogang.ac.kr)
- 기한: 15/06/16
- 프로젝트 내용: objectA,B (A/B는 label)를 받아 학습하고 test_set의 label을 예측


### 프로젝트 버전 정보
	- python 3.4.2
	- scipy 0.15.1
	- scikit-learn 0.16.1
	- numpy 1.9.2

### 폴더 및 프로젝트 설명

#### Datang
샘플 데이터 및 테스트 데이터 폴더

- **objectA.mat**: label A에 대한 데이터가 저장
- **objectB.mat**: label B에 대한 데이터가 저장
- **test_set.mat**: test_set에 대한 데이터가 저장
- **DataModule.py**: Data를 python 형식으로 수집하고 변환하는 함수가 저장
- **PredictionModule.py**: 데이터를 읽고 학습한 뒤 예측치를 반환하는 연산 기록



#### Prediction Module
총 4종류의 기계학습 알고리즘을 이용하여 학습을 수행하였음

1. Random Forest Classifier
2. Linear SVC (Support Vector Classifier)
3. SVC (Support Vector Classifier)
4. K-NN (K-Nearest Neighbor Classifier)

#### 학습결과
학습 및 추정은 4자 다수로 확신하여 답안을 제출하기로 하였으며, test_set 48종에 대한 각각의 추정치는 아래와 같음.

1. RANDOM FOREST CLASSIFIER
	
		[1 0 0 1 1 1 0 1 0 1 0 1 0 0 1 1 0 0 1 0 1 1 0 0 0 1 0 0 0 1 1 1 0 0 1 1 1 1 0 0 0 1 0 1 1 1 0 0]

2. LINEAR SUPPORT VECTOR MACHINE
	
		[1 0 0 1 1 1 0 1 0 1 0 1 0 0 1 1 0 0 1 0 1 1 0 0 0 1 0 0 0 1 1 1 0 0 1 1 1 1 0 0 0 1 0 1 1 1 0 0]

3. SUPPORT VECTOR CLASSIFIER

		[1 1 1 1 1 0 0 0 0 1 1 1 0 0 1 1 1 1 1 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 0 0 1 1 1 0 1 0 1 1 1 0 0]

4. K-NEAREST NEIGHBOR CLASSIFIER

		[0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 1 0 0 0 0 1 1 0 0 0 0 0 0 0 1 1 0 0 0 1 0 0 1 0 0 0 1 0 1 1 1 0 0]
		
5. 답안제출

		답안은 각 알고리즘별로 수집한 뒤, RF > SVC > L-SVC > K-NN 순으로 우선순위를 취합하여 다수의 라벨을 정답으로 측정하여 제출하였음.