'''
Train Dataset 전처리 과정을 정리하고 테스트 하는 목적으로 작성된 공간.

'''
import torch
import pandas as pd
import Preprocessing as prp
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader, RandomSampler


kium = pd.read_csv(r'.\TrainCopySet.csv')
raw_df = pd.DataFrame(kium)

# 50개씩 검사
tdf = raw_df.iloc[150:170].copy()
#tdf = raw_df.copy()

# patt = r'[a-z][가-힣]|patient'
# tdf = raw_df.loc[
#     raw_df['Findings'].str.contains(patt, regex=True, na=False)
# ].copy()

####################################################################################
# Nan(결측값) 처리.
#tdf['Conclusion'] = tdf['Conclusion'].fillna('conclusion unremarkable')
#tdf['Findings'] = tdf['Findings'].fillna('finding unremarkable')

prp.empty_to_missing(tdf)
print('결측값 처리 완료')
#####################################################################################

prp.sent_tokenizing(tdf)
print('문장 토큰화 작업 처리')

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

prp.unnecessary_preprocessing(tdf)
print('불필요 용어 처리 완료')

prp.special_token_preprocessing(tdf)
print('special token 추가 작업 완료')

# 'tokens' Column 생성되는 위치.
prp.word_tokenizing(tdf)
print('단어 토큰화 작업 완료')

max_token_size = prp.stopword_removal(tdf)
print(f'불용어 토큰 삭제 처리 완료 :  {max_token_size}')

# 'input_ids', 'padd_ids', 'att_ids' Column 생성되는 위치.
prp.embedding(tdf)
print('단어 임베딩 및 attention mask 생성 완료')


idx_list = tdf.index.tolist()
for idx in idx_list:
    print(raw_df.loc[idx, 'Findings'])
    print(raw_df.loc[idx, 'Conclusion'])
    print('-------------------------------------------------------------------------------')
    print(tdf.context.loc[idx])
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(tdf.tokens.loc[idx])
    print(tdf.input_ids.loc[idx])
    print(tdf.padd_ids.loc[idx])
    print('###############################################################################')
    print()

print(max_token_size)
print(tdf['padd_ids'].apply(len).max())


batch_size = 16  # 또는 32 등 원하는 배치 크기

# 레이블, input 데이터, 마스킹 값
encoder = LabelEncoder()
train_labels = encoder.fit_transform(tdf['AcuteInfarction'])

train_tensor = torch.tensor(tdf['padd_ids'].tolist())
train_mask = torch.tensor(tdf['att_ids'].tolist())
train_labels = torch.tensor(train_labels)

#9. DataLoader 생성
train_dataset = TensorDataset(train_tensor, train_mask, train_labels)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)


prp.training(train_dataloader)


# for idx in index_list:
#     print(tdf['Findings'].iloc[idx], pd.isna(tdf['Findings'].iloc[idx]))
#     print()
#     print(tdf['Conclusion'].iloc[idx])
#     print()
#     print(tdf['context'].iloc[idx])
#     print('####################################################################################')

