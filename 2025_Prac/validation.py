'''
- Preprocessing.py에서 전처리된 데이터로 학습한 model을 불러옴.
- 검증(테스트) 데이터셋은 'ValidationSet_2차.csv'이므로 이를 통해 정확도를 계산
- C-statistics (AUROC)을 기반으로 정확도를 검증.
'''

import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from transformers import BertForSequenceClassification
import Preprocessing as pr
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt


model = BertForSequenceClassification.from_pretrained("../../saved_bert_model_3")
device = pr.Checking_cuda()
model.to(device)

kiumSet = pd.read_csv(r'.\ValidationSet.csv')
df = pd.DataFrame(kiumSet)
pre_df = pd.DataFrame(columns=['finding', 'conclusion', 'class'])


pr.show_info(df)

pr.empty_to_missing(df)
raw_find, after_find = pr.Findings_Preprocessing(df, pre_df)
raw_conc, after_conc = pr.Conclusion_Preprocessing(df, pre_df)
pr.Acute_Classification(df, pre_df)

pre_df['context'] = pre_df['finding'] + pre_df['conclusion']
pre_df = pre_df.drop(columns=['finding', 'conclusion'])

# for idx, c in enumerate(pre_df['context']):
#     if idx == 30: break
#     print(c)
#     print()


test_inputs = pr.word_sequencing(pre_df)
test_masks = pr.attention_masking(test_inputs)

encoder = LabelEncoder()
test_labels = encoder.fit_transform(pre_df['class'])


test_inputs = torch.tensor(test_inputs).long()
test_labels = torch.tensor(test_labels).long()
test_masks = torch.tensor(test_masks).long()

batch_size = 32  # 또는 32 등 원하는 배치 크기


# 평가 모드
model.eval()

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)  # 순차적으로 순회 (정답 순서 보장)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

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

    # GPU를 사용했다면 예측 결과와 정답지 CUDA Tensor를 CPU로 변환
    if device.type == 'cuda':
        preds = preds.to('cpu').numpy()
        labels = labels.to('cpu').numpy()

    # 두 값중 큰 값을 예측 결과로 두고, 펼침
    pred_flat = np.argmax(preds, axis=1).flatten()
    pred_list.extend(pred_flat)
    # 정답 라벨 (비교용)
    label_list.extend(labels)

# 예측값(pred_list)과 정답지(label_list)를 AUROC 계산
fpr, tpr, thresholds = roc_curve(pred_list, label_list)
# AUC 면적
print(f"정확도(Accuracy): {round(roc_auc_score(pred_list, label_list),6)}")

# ROC 곡선
# plt.plot(fpr, tpr)
# plt.xlabel('FP Rate')
# plt.ylabel('TP Rate')
# plt.show()