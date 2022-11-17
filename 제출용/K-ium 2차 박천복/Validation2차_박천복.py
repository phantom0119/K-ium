# 2차 제출 목적용 Validation 프로그램
# 동일 경로 내에 'Validation박천복.py' 파일이 있어야 합니다.
# 1차 제출한 파일의 함수를 수정 없이 사용했습니다. (import Validation박천복)

from sklearn.metrics import roc_auc_score  # AUROC
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import torch.nn.functional as functional
import numpy as np
import pandas as pd
import torch, nltk
import Validation박천복 as valifunc

nltk.download('punkt')     # 구두점 분리가 학습된 모델
nltk.download('stopwords') # 불용어 사전

'''
2차 데이터셋 이름 = 'ValidationSet_2차.csv' 
'''
PATH = '.\ValidationSet_2차.csv'  # 테스트 파일 경로 기입
MPATH = '.\model_save_CPU.pt'  # 사용할 모델 경로 (같은 디렉터리에 있다고 가정)
# BERT Tokenizer 최대 길이(MAX_LEN) = 512
MAX_LEN = 512


if __name__ == '__main__':
    #settime = time.time()  # 처리 시간을 계산할 경우

    # 검증에 사용할 데이터를 엑셀(csv)에서 가져온다고 가정
    kiumSet = pd.read_csv(PATH)
    df = pd.DataFrame(kiumSet)

    # Conclusion은 \n 문자가 있으므로 이를 처리하고 진행
    # 혹시 모르니 columns 전부 명시적 변경
    df.columns = ['Findings', 'Conclusion', 'AcuteInfarction']

    # 1. 결측값(NaN) 전처리
    #valifunc.empty_to_missing(df)
    df.fillna('', inplace=True)

    # 2. 불필요한 텍스트 데이터 전처리
    valifunc.Pretreatment(df)

    # 3. 정제한 데이터를 BERT에 적용하기 위한 문장 토큰화
    test_sentences = valifunc.sent_tokenizing(df)

    # 4. 문장 토큰을 단어 토큰으로 세분화 --> 단어 시퀀스 생성 (BERT Tokenizing)
    test_inputs = valifunc.BERT_Tokenizing_Model(test_sentences)

    # Attention Mask
    test_masks = valifunc.Attention_Masking(test_inputs)

    # 정답지
    test_labels = df['AcuteInfarction'].values

    # 5. 검증에 사용할 device와 model 구조 설정
    '''
        학습된 모델은 cpu로 기반으로 만들었기 때문에 GPU를 사용해 검증할 경우 변환 필요.
    '''
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = torch.load(MPATH, map_location=f"cuda:{torch.cuda.current_device()}")
        model.to(device)  # Torch Tensor  -->  CUDA Tensor 변환
    else:
        device = torch.device("cpu")
        model = torch.load(MPATH)

    # GPU 적용이 안 되는 전제가 있다면 아래 주석 코드 사용 (직접 명시)
    # device = torch.device("cpu")

    # 6. 파이토치로 검증하기 위한 텐서 설정
    test_inputs = torch.tensor(test_inputs)
    test_masks = torch.tensor(test_masks)
    test_labels = torch.tensor(test_labels)

    # 7. 텐서 Dataset 설정
    '''
      Dataset은 torch.utils.data.Dataset 의 하위 클래스.
      DataLoader는 Dataset을 샘플에 쉽게 접근할 수 있도록 순회 가능한 객체(iterable)로 담는다.
      RandomSampler는 Dataset을 섞음.
    '''
    batch_size = 32  # batch
    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)



    '''
    ##############################################
        8. 모델 테스팅 진행, 출력 요구사항 반영
    ##############################################
    '''
    data_len = len(test_dataloader)

    model.eval()
    # pred_list: 예측결과, label_list: 정답지
    pred_list = []; label_list = []

    for idx, batches in enumerate(test_dataloader):
        # batch 정보를 device에 넣음
        batch = tuple(batch_val.to(device) for batch_val in batches)
        # batch에서 input_ids, attention_mask, label 추출
        input_ids, input_mask, labels = batch

        # Grad 계산 안함
        with torch.no_grad():
            # (input_ids에 대한 예측 결과 [0일 수치, 1일 수치]로 표현 -> logits[0]
            logits = model(input_ids, token_type_ids=None, attention_mask=input_mask)

        preds = logits[0]
        proba = functional.softmax(preds, dim=1) # 소프트맥스 함수로 [0,1] 범위 표현

        # GPU를 사용했다면 예측 결과와 정답지 CUDA Tensor를 CPU로 변환
        if device.type == 'cuda':
            preds = preds.to('cpu').numpy()
            labels = labels.to('cpu').numpy()
            proba = proba.to('cpu').numpy()

        '''뇌졸중 예측 결과 출력 [1로 판정한 확률, 0으로 판정한 확률]'''
        for case in proba:
            # [0,1] 범위 소프트맥스 출력, (뇌졸중 확률) (뇌졸중 아닌 확률)
            print(f"{round(float(case[1]), 4)} {round(float(case[0]), 4)}")


        # 두 값중 큰 값을 예측 결과로 두고, 펼침
        pred_flat = np.argmax(preds, axis=1).flatten()
        pred_list.extend(pred_flat)
        # 정답 라벨 (비교용)
        label_list.extend(labels)

    # AUC 면적
    print(f"정확도(Accuracy): {round(roc_auc_score(pred_list, label_list), 6)}")


