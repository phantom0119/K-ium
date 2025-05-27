'''
- Preprocessing.py에서 전처리된 데이터로 학습한 model을 불러옴.
- 검증(테스트) 데이터셋은 'ValidationSet_2차.csv'이므로 이를 통해 정확도를 계산
- C-statistics (AUROC)을 기반으로 정확도를 검증.
'''

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from transformers import BertForSequenceClassification
import Preprocessing as prp
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler


model = BertForSequenceClassification.from_pretrained("../../saved_bert_model_3")

kiumSet = pd.read_csv(r'.\ValidationSet.csv')
df = pd.DataFrame(kiumSet)

vdf = df.copy()

prp.empty_to_missing(vdf)
print('결측값 처리 완료')
#####################################################################################

prp.sent_tokenizing(vdf)
print('문장 토큰화 작업 처리')

prp.lobe_preprocessing(vdf)
print('lobe 전처리 작업 완료')

prp.semi_preprocessing(vdf)
print('세부 전처리 작업 완료')

prp.medical_words_preprocessing(vdf)
print('의학 용어 전처리 작업 완료')

prp.cardi_ordinal_preprocessing(vdf)
print('순서/수량 값의 전처리 작업 완료')

prp.demention_preprocessing(vdf)
print('크기 데이터 전처리 작업 완료')

prp.pos_neg_preprocessing(vdf)
print('긍정/부정 값 전처리 작업 완료')

prp.unnecessary_preprocessing(vdf)
print('불필요 용어 처리 완료')

prp.special_token_preprocessing(vdf)
print('special token 추가 작업 완료')

# 'tokens' Column 생성되는 위치.
prp.word_tokenizing(vdf)
print('단어 토큰화 작업 완료')

max_token_size = prp.stopword_removal(vdf)
print(f'불용어 토큰 삭제 처리 완료 :  {max_token_size}')

# 'input_ids', 'padd_ids', 'att_ids' Column 생성되는 위치.
prp.embedding(vdf)
print('단어 임베딩 및 attention mask 생성 완료')


batch_size = 16  # 또는 32 등 원하는 배치 크기

# 레이블, input 데이터, 마스킹 값
encoder = LabelEncoder()
test_labels = encoder.fit_transform(vdf['AcuteInfarction'])

test_tensor = torch.tensor(vdf['padd_ids'].tolist())
test_mask = torch.tensor(vdf['att_ids'].tolist())
test_labels = torch.tensor(test_labels)

#9. DataLoader 생성
test_dataset = TensorDataset(test_tensor, test_mask, test_labels)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

prp.validating(model, test_dataloader)

'''  4번 모델 결과
                precision    recall  f1-score   support

           0       1.00      1.00      1.00      2423
           1       0.96      0.95      0.96       230

    accuracy                           0.99      2653
   macro avg       0.98      0.97      0.98      2653
weighted avg       0.99      0.99      0.99      2653

정확도(Accuracy): 0.97423
'''


idx_list = vdf.index.tolist()
for idx in idx_list:
    print(df.loc[idx, 'Findings'])
    print(df.loc[idx, 'Conclusion'])
    print('-------------------------------------------------------------------------------')
    print(vdf.context.loc[idx])
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(vdf.tokens.loc[idx])
    print(vdf.input_ids.loc[idx])
    print(vdf.padd_ids.loc[idx])
    print('###############################################################################')
    print()


