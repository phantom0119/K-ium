'''
BERT tokenizer로 토큰화한 결과를 추가 처리 없이 직접 학습에 사용할 경우 정확도를 측정한다.
'''
import pandas as pd
import numpy as np
import torch, gc
import nltk
import re
from nltk.tokenize import sent_tokenize
import Preprocessing as prp
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertForSequenceClassification, BertConfig
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from torch.optim import AdamW

nltk.download('punkt')      # 구두점 분리가 학습된 모델
nltk.download('stopwords')  # 불용어 사전
# 토큰화된 결과에서 유의미한 토큰이 가장 많았을 때의 개수
max_token_size = 0

stopwords = ('and', 'at', 'a', 'an', 'as', 'are', 'b', 'in', 'to', 'the', 'of', 'or',
             'm', 's', '~', '"', '-',
             '.', ',', ':', '(', ')', '→', '[', ']', '/', '*', '=', '+', "'", '&', '#', '?', ';')


def training(model : BertForSequenceClassification,
             device : torch.device,
             train_dataloader : DataLoader):
    gc.collect()
    torch.cuda.empty_cache()

    # 모델을 학습 모드로 두고 진행.
    model.train()
    # 모델을 gpu에 담기.
    model.to(device)
    # 토크나이저 단어 사전에 사용자 추가된 것이 있으므로 개수 반영.
    model.resize_token_embeddings(len(tokenizer_bert))
    # 옵티마이저
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    # 모델 에폭수
    epochs = 4
    # 총 훈련 스탭 = 배치 반복 횟수 * 에폭수
    total_steps = len(train_dataloader) * epochs
    # 스케줄러 생성
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps= int(0.1 * total_steps),
                                                num_training_steps=total_steps) # 또는 0

    # 에폭수만큼 배치 학습 반복 (조기 종료 추가)
    for epoch in range(epochs):
        total_loss = 0          # 평균 손실값 계산용

        for step, batch in enumerate(train_dataloader):
            # 학습에 사용할 train_dataloader의 각 항목(inputs, attention, label)을 device에 담기.
            # 배치 사이즈에 맞는 데이터 묶음이 담겨 있다.
            b_input_ids = batch[0].long().to(device)
            b_attention_mask = batch[1].long().to(device)
            b_labels = batch[2].long().to(device)

            # 기울기 초기화
            optimizer.zero_grad()

            # Forward 연산 결과 ( 예측-logits, 손실-loss 계산 )
            outputs = model(
                input_ids = b_input_ids,
                attention_mask = b_attention_mask,
                labels = b_labels
            )

            # 손실 (역전파에 사용)
            loss = outputs.loss
            total_loss += loss.item()
            # 예측 (softmax(), argmax() 등으로 class 결정)
            logits = outputs.logits

            # Backward
            loss.backward()
            # 그래디언트 클리핑 (기울기 폭주 방지)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Model의 Parameter Update
            optimizer.step()
            # 학습률(Learning Rate) 동적 조율
            scheduler.step()

            if step % 10 == 0:
                print(f"[Epoch {epoch + 1}] Step {step} - Loss: {loss.item():.4f}")

        # 평균 로스 계산
        avg_train_loss = total_loss / len(train_dataloader)
        print(f'평균 loss = {avg_train_loss}')


    # 학습 완료한 모델 저장
    model.save_pretrained(model_save_path)
    tokenizer_bert.save_pretrained(model_save_path)



def validation(test_dataloader):
    gc.collect()
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model2 = BertForSequenceClassification.from_pretrained(model_save_path)
    model2.to(device)
    model2.eval()

    predictions = []    # 예측값
    true_labels = []    # 실제값

    with torch.no_grad():
        for batch in test_dataloader:
            b_input_ids = batch[0].long().to(device)
            b_attention_mask = batch[1].long().to(device)
            b_labels = batch[2].long().to(device)

            outputs = model2(
                input_ids=b_input_ids,
                attention_mask=b_attention_mask
            )

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = b_labels.cpu().numpy()

            predictions.extend(preds)
            true_labels.extend(labels)

    print(classification_report(true_labels, predictions, digits=4))
    print("\n[정확도]")
    print("Accuracy:", accuracy_score(true_labels, predictions))
    cstat = round(roc_auc_score(predictions, true_labels), 6)
    print(f"AUC: {cstat}")

    return cstat


if __name__ == '__main__':
    kiumSet = pd.read_csv(r'.\TrainCopySet.csv')
    validSet = pd.read_csv(r'.\ValidationSet.csv')

    kdf = pd.DataFrame(kiumSet)
    vdf = pd.DataFrame(validSet)

    # 학습/테스트 DataFrame
    tdf = kdf.copy()
    vdf = vdf.copy()

    # mim = 2000
    # lim = 2050
    # tdf = tdf.iloc[mim:lim]

    # 모델 저장/불러오기 경로
    model_save_path = '../../saved_bert_model_5'

    # tokenizer ( PubMed 초록(abstract)만을 사용하여 처음부터 사전학습한 모델 )
    tokenizer_bert = BertTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')


    #1. 결측값 처리
    prp.empty_to_missing(tdf)
    prp.empty_to_missing(vdf)


    #2. Findings + Conclusion 후, 문장 토큰화로 [CLS], [SEP] 토큰 추가하기.
    train_sentences = sent_tokenizing(tdf)
    test_sentences = sent_tokenizing(vdf)

    """
    [CLS] Clinical information : Lung cancer. [SEP] Axial T1WI, sagittal T1WI, axial T2WI, axial FLAIR, axial T2* GRE image 획득하였으며 조영증강을 시행함. [SEP] Decreased size of heterogeneously enhancing large mass in left basal ganglia and frontal lobe(2.8x2.7cm -> 2.3x2.2cm). [SEP] -- With hemorrhagic transformation. [SEP] DDx. [SEP] 1) metastasis
           2) malignancy such as GBM [SEP]
    """

    #3. 토큰이 추가된 문장을 단어 토큰으로 생성 --> 단어 Sequence 생성.
    train_inputs = BERT_Tokenizing_Model(train_sentences)
    test_inputs = BERT_Tokenizing_Model(test_sentences)

    #4. 정답지
    encoder = LabelEncoder()
    train_labels = encoder.fit_transform(tdf['AcuteInfarction'])
    test_labels = encoder.fit_transform(vdf['AcuteInfarction'])

    #5. inputs에 대한 Attention mask 생성
    train_masks = prp.attention_masking(train_inputs)
    test_masks = prp.attention_masking(test_inputs)


    # for idx in range(0,50):
    #     #if '[UNK]' in train_sentences[idx]:
    #     print(tdf.iloc[idx].Findings + ' ' + tdf.iloc[idx].Conclusion)
    #     print()
    #     print(train_sentences[idx])
    #     print()
    #     print(tokenizer_bert.convert_ids_to_tokens(train_inputs[idx]))
    #     print('####################################################################################')


    #6. 학습에 필요한 gpu 활성화
    device = prp.Checking_cuda()


    #7. BERT 학습 모델.
    #먼저 구성 객체 설정
    config = BertConfig.from_pretrained(
        'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
        num_labels=2
    )

    # 분류용 헤드를 수동으로 생성
    model = BertForSequenceClassification.from_pretrained(
        'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
        config=config
    )

    batch_size = 32  # 또는 32 등 원하는 배치 크기

    #8. 학습/검증을 위한 torch tensor 변환.
    train_inputs = torch.tensor(train_inputs).long()
    train_labels = torch.tensor(train_labels).long()
    train_masks = torch.tensor(train_masks).long()
    test_inputs = torch.tensor(test_inputs).long()
    test_labels = torch.tensor(test_labels).long()
    test_masks = torch.tensor(test_masks).long()

    #9. DataLoader 생성
    train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(test_data)  # 순차적으로 순회 (정답 순서 보장)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    print(max_token_size)

    #10. 모델 학습 진행.
    training(model, device, train_dataloader)

    #11. 모델 검증(테스트) 진행.
    c_statistic = validation(test_dataloader)



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

'''
4차 테스트
epoch=4 가 좋은 것 같다.
lr = 1e 말고 2e-5 등도 시도해본다.
기존 학습된 모델에 거듭 학습한 모델을 _4에만 저장한다.
2e-5 는 0.96 수준.  크게 개선되지 않음
                precision    recall  f1-score   support

           0     0.9922    0.9934    0.9928      2425
           1     0.9289    0.9167    0.9227       228

    accuracy                         0.9868      2653
   macro avg     0.9605    0.9550    0.9578      2653
weighted avg     0.9867    0.9868    0.9868      2653

[정확도]
Accuracy: 0.9868073878627969
AUC: 0.960532

'''