'''
RandomForest 적용 텍스트 분류.
'''
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report


kiumSet = pd.read_csv(r'.\TrainCopySet.csv')
validSet = pd.read_csv(r'.\ValidationSet.csv')

tdf = pd.DataFrame(kiumSet)
vdf = pd.DataFrame(validSet)

tdf.fillna('no-text', inplace=True)
vdf.fillna('no-text', inplace=True)

tdf['context'] = tdf['Findings'] + tdf['Conclusion']
vdf['context'] = vdf['Findings'] + vdf['Conclusion']

tdf.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 6190 entries, 0 to 6189
Data columns (total 3 columns):
 #   Column           Non-Null Count  Dtype 
---  ------           --------------  ----- 
 0   Findings         4814 non-null   object
 1   Conclusion       6156 non-null   object
 2   AcuteInfarction  6190 non-null   int64 
dtypes: int64(1), object(2)
memory usage: 145.2+ KB
'''

#vdf.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2653 entries, 0 to 2652
Data columns (total 3 columns):
 #   Column           Non-Null Count  Dtype 
---  ------           --------------  ----- 
 0   Findings         2096 non-null   object
 1   Conclusion       2626 non-null   object
 2   AcuteInfarction  2653 non-null   int64 
dtypes: int64(1), object(2)
memory usage: 62.3+ KB
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", num_labels=2)
model.to(device)
model.eval()

# BERT 임베딩 추출 함수 (CLS 토큰 사용)
def get_bert_embedding(text):
    inputs = tokenizer(text
                       , return_tensors="pt"
                       , truncation=True
                       , padding='max_length'
                       , max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding.squeeze(0).cpu().numpy()

# 모든 보고서에 대해 임베딩 벡터 생성
train_x = np.vstack([get_bert_embedding(text) for text in tqdm(tdf['context'])])
train_y = tdf['AcuteInfarction'].values

test_x = np.vstack([get_bert_embedding(text) for text in tqdm(vdf['context'])])
test_y = vdf['AcuteInfarction'].values

rfcl = RandomForestClassifier(random_state=42)

rfcl.fit(train_x, train_y)
y_pred = rfcl.predict(test_x)

print(classification_report(y_pred, test_y))
print(roc_auc_score(y_pred, test_y))

"""
100%|██████████| 6190/6190 [05:25<00:00, 18.99it/s]
100%|██████████| 2653/2653 [02:26<00:00, 18.14it/s]
              precision    recall  f1-score   support

           0       1.00      0.95      0.97      2555
           1       0.39      0.91      0.55        98

    accuracy                           0.94      2653
   macro avg       0.69      0.93      0.76      2653
weighted avg       0.97      0.94      0.95      2653

0.9268800670953312
"""

