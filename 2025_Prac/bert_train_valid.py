'''
BERT tokenizer로 토큰화한 결과를 추가 처리 없이 직접 학습에 사용할 경우 정확도를 측정한다.
'''
import pandas as pd
import torch
import Preprocessing as prp
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler

if __name__ == '__main__':
    kiumSet = pd.read_csv(r'.\TrainCopySet.csv')
    validSet = pd.read_csv(r'.\ValidationSet.csv')

    # 학습/테스트 DataFrame
    mdf = pd.DataFrame(kiumSet)
    vdf = pd.DataFrame(validSet)

    #tdf = mdf.loc[30:60].copy()
    # tflist = tdf['Findings'].tolist()
    # tconlist = tdf['Conclusion'].tolist()
    tdf = mdf.copy()


    #1. 결측값 처리
    #prp.empty_to_missing(tdf)
    prp.empty_to_missing(vdf)


    #2. Findings + Conclusion 후, 문장 토큰화로 [CLS], [SEP] 토큰 추가하기.
    #train_sentences = prp.sent_tokenizing(tdf)
    test_sentences = prp.sent_tokenizing(vdf)


    #3. 토큰이 추가된 문장을 단어 토큰으로 생성 --> 단어 Sequence 생성.
    # Flag로 Training 시에는 새 토크나이저로 작업 (0),
    # Validating 시에는 저장된 토크나이저로 작업 (1)
    #train_inputs, train_sent = prp.tokenizing(train_sentences, 0)
    test_inputs, test_sent = prp.tokenizing(test_sentences, 1 )

    #4. 정답지
    encoder = LabelEncoder()
    #train_labels = encoder.fit_transform(tdf['AcuteInfarction'])
    test_labels = encoder.fit_transform(vdf['AcuteInfarction'])

    #5. inputs에 대한 Attention mask 생성
    #train_masks = prp.attention_masking(train_inputs)
    test_masks = prp.attention_masking(test_inputs)

    batch_size = 16  # 또는 32 등 원하는 배치 크기

    #8. 학습/검증을 위한 torch tensor 변환.
    # train_inputs = torch.tensor(train_inputs).long()
    # train_labels = torch.tensor(train_labels).long()
    # train_masks = torch.tensor(train_masks).long()
    test_inputs = torch.tensor(test_inputs).long()
    test_labels = torch.tensor(test_labels).long()
    test_masks = torch.tensor(test_masks).long()

    #9. DataLoader 생성
    # train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
    # train_sampler = RandomSampler(train_dataset)
    # train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(test_data)  # 순차적으로 순회 (정답 순서 보장)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    #10. 모델 학습 진행.
    #prp.training(train_dataloader)

    #11. 모델 검증(테스트) 진행.
    c_statistic = prp.validation(test_dataloader)


'''
한글 토큰을 별도 토크나이저로 추출 후 학습
1차 테스트 결과
                precision    recall  f1-score   support

           0     0.9942    0.9946    0.9944      2425
           1     0.9427    0.9386    0.9407       228

    accuracy                         0.9898      2653
   macro avg     0.9685    0.9666    0.9675      2653
weighted avg     0.9898    0.9898    0.9898      2653

[정확도]
Accuracy: 0.9898228420655861
AUC: 0.96848
'''

'''
한글 토큰에서 '##'으로 분리된 토큰을들 결합 후 학습
2차 테스트 결과
                precision    recall  f1-score   support

           0     0.9955    0.9951    0.9953      2425
           1     0.9476    0.9518    0.9497       228

    accuracy                         0.9913      2653
   macro avg     0.9715    0.9734    0.9725      2653
weighted avg     0.9913    0.9913    0.9913      2653

[정확도]
Accuracy: 0.9913305691669808
AUC: 0.97153
'''

'''
한글 + 영어 모두 '##'으로 분리된 토큰들을 결합 후 학습
3차 테스트 결과
                precision    recall  f1-score   support

           0     0.9212    0.9984    0.9582      2425
           1     0.8400    0.0921    0.1660       228

    accuracy                         0.9205      2653
   macro avg     0.8806    0.5452    0.5621      2653
weighted avg     0.9143    0.9205    0.8902      2653

[정확도]
Accuracy: 0.9204673954014323
AUC: 0.880616
'''

"""  1번 모델 정확도
              precision    recall  f1-score   support

           0     0.9946    0.9963    0.9955      2425
           1     0.9598    0.9430    0.9513       228

    accuracy                         0.9917      2653
   macro avg     0.9772    0.9696    0.9734      2653
weighted avg     0.9917    0.9917    0.9917      2653


[정확도]
Accuracy: 0.9917075009423294
AUC: 0.977235

"""