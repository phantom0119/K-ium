'''
Train Dataset 전처리 과정을 정리하고 테스트 하는 목적으로 작성된 공간.

'''
import torch
import re
import pandas as pd
import Preprocessing as prp
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import BertTokenizer


#kium = pd.read_csv(r'.\TrainCopySet.csv')
kium = pd.read_csv(r'.\ValidationSet.csv')
raw_df = pd.DataFrame(kium)


#tdf = raw_df.iloc[360:390].copy()
#tdf = raw_df[raw_df['AcuteInfarction']==1].copy()


# patt = r'\(?\d{4}([.\- ]*\d{1,2}[.\- ]*\d{1,2}[.\-), ]*)+'
# tdf = raw_df.loc[raw_df['Conclusion'].str.contains(
#     patt, regex=True, na=False
# )].copy()

prp.empty_to_missing(tdf)
print('결측값 처리 완료')
#####################################################################################

prp.sent_tokenizing(tdf)
print('문장 토큰화 작업 처리')

# idx_list = tdf.index.tolist()
# for idx in idx_list:
#     #print('전처리 전 Conclusion Text')
#     print(raw_df['Findings'].loc[idx])
#     print(raw_df['Conclusion'].loc[idx])
#     print(raw_df['AcuteInfarction'].loc[idx])
#     #print()
#     #print('전처리 후 Conclusion Text')
#     print(tdf['context'].loc[idx])

prp.lobe_preprocessing(tdf)
print('lobe 전처리 작업 완료')

prp.semi_preprocessing(tdf)
print('세부 전처리 작업 완료')

prp.medical_words_preprocessing(tdf)
print('의학 용어 전처리 작업 완료')

prp.cardi_ordinal_preprocessing(tdf)
print('순서/수량 값의 전처리 작업 완료')

prp.demention_preprocessing(tdf)
print('크기 데이터 전처리 작업 완료')

prp.pos_neg_preprocessing(tdf)
print('긍정/부정 값 전처리 작업 완료')

prp.special_token_preprocessing(tdf)
print('special token 추가 작업 완료')

# 'tokens' Column 생성되는 위치.
prp.word_tokenizing(tdf)
print('단어 토큰화 작업 완료')

max_token_size = prp.stopword_removal(tdf)

# 'input_ids', 'padd_ids', 'att_ids' Column 생성되는 위치.
prp.embedding(tdf)
print('단어 임베딩 및 attention mask 생성 완료')

idx_list = tdf.index.tolist()
for idx in idx_list:
    #print('전처리 전 Conclusion Text')
    print(raw_df['Findings'].loc[idx])
    print(raw_df['Conclusion'].loc[idx])
    print(raw_df['AcuteInfarction'].loc[idx])
    #print()
    #print('전처리 후 Conclusion Text')
    print(tdf['context'].loc[idx])
    print('-------------------------------------------------------------------------------------------')
    print(tdf['tokens'].loc[idx])
    print('##########################################################################################')


