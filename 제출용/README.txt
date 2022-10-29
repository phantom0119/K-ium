## 파일 구성 ##
1. C-statistics.txt  (70%의 데이터셋 검증 결과)
2. Training(박천복).py (모형 구축에 사용한 코드)
3. Validation(박천복).py (모형 검증에 사용할 코드)
4. model_save_CPU.pt  (Training 결과로 저장한 모형)

-----------------------------------------------------------------------
1차에 사용한 70% 데이터셋,
그리고 2차인 30% 데이터셋 검증을 위해
Validation(박천복).py 코드를 사용합니다.

Validation(박천복).py 코드 안에 주석으로 사용법 작성했습니다.
검증에 사용할 데이터셋은 PATH = '.\Dataset.csv'로 경로 명시했습니다.

검증에 사용할 데이터셋은 csv 파일이고,
그 파일명이 Dataset으로 되어있다고 가정했기 때문입니다.
------------------------------------------------------------------------

## 사용한 환경 및 라이브러리 버전 정보 ## 
  conda = 22.9.0 (python 3.8.5)
  tensorflow(keras) = 2.10.0   Apache 2.0
  nltk = 3.7    Apache 2.0
  transformers = 4.22.2  Apache 2.0
  scikit-learn = 1.1.1  BSD license
  torch = 1.12.1        BSD license
  numpy = 1.23.1      BSD license
  pandas = 1.5.0       BSD license
  matplotlib = 3.5.2   BSD license


