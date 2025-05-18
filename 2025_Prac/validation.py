'''
- Preprocessing.py에서 전처리된 데이터로 학습한 model을 불러옴.
- 검증(테스트) 데이터셋은 'ValidationSet_2차.csv'이므로 이를 통해 정확도를 계산
- C-statistics (AUROC)을 기반으로 정확도를 검증.
'''

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from transformers import BertForSequenceClassification
import Preprocessing as pr
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from sklearn.metrics import roc_auc_score


model = BertForSequenceClassification.from_pretrained("saved_bert_model_2")
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

batch_size = 16  # 또는 32 등 원하는 배치 크기


# 평가 모드
model.eval()

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)  # 순차적으로 순회 (정답 순서 보장)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)


predictions = []
true_labels = []

with torch.no_grad():
    for batch in test_dataloader:
        b_input_ids = batch[0].to(device).long()
        b_attention_mask = batch[1].to(device).long()
        b_labels = batch[2].to(device).long()

        outputs = model(
            input_ids=b_input_ids,
            attention_mask=b_attention_mask
        )

        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        labels = b_labels.cpu().numpy()

        predictions.extend(preds)
        true_labels.extend(labels)

# 3. 예측 결과 비교
for pred, true in zip(predictions, true_labels):
    print(f"예측: {pred}, 실제: {true}")


# 선택적으로 성능 평가
from sklearn.metrics import classification_report, accuracy_score

print(classification_report(true_labels, predictions, digits=4))
print("\n[정확도]")
print("Accuracy:", accuracy_score(true_labels, predictions))

print('\n[C-statistic]')
print('AUC: ', roc_auc_score(true_labels, predictions))