'''
BERT tokenizer로 토큰화한 결과를 추가 처리 없이 직접 학습에 사용할 경우 정확도를 측정한다.
'''
import pandas as pd
import numpy as np
import torch
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
from tqdm import tqdm
import pytest

# 토큰화된 결과에서 유의미한 토큰이 가장 많았을 때의 개수
max_token_size = 0

stopwords = ('and', 'at', 'a', 'an', 'as', 'are', 'b', 'in', 'to', 'the', 'of', 'or',
             'm', 's', '~', '"', '-',
             '.', ',', ':', '(', ')', '→', '[', ']', '/', '*', '=', '+', "'", '&', '#', '?', ';')


def semi_preprocessing(text : str) -> str:
    ##  4. 'ms'를 의미하는 수치 데이터 전처리.
    tmp = text
    text = re.sub(r'\b(\d+)\s*(MS|ms)\b|TE\s+(\d+)'
                   , lambda m: (
            f"ms-{m.group(1)}" if m.group(1) else  # \1: 숫자 (ms)
            f"te ms-{m.group(3)}" if m.group(3) else ""  # TE 144 등의 ms없는 표현
        ), text)

    ##  5. % = percent 값 전처리.
    text = re.sub(r'(\d+)\s*\%', r' percent-\1 ', text)  # 10%  --> percent-10

    ##  6. Cho/NAA 수치 데이터 전처리.
    #  Cho/Cr = 2.29,  Cho/NAA = 1.14
    matches = re.findall(
        r'\(?(Cho\-NAA|Cho\-Cr|[eE]vans\-index|[cC]allosal\-angle)(?:\s|\,|\=|increased)+((?:\d+(?:\.\d+)?(?:[, ;.]+)?)+)(?:\)?\.?| at|\()',
        text)

    if matches:
        for grplist in matches:
            # grplist: tuple → ('Cho-NAA', '0.85 0.79 0.90')
            grp_values = grplist[1]  # 수치 문자열 전체
            values = re.findall(r"\d+(?:\.\d+)?", grp_values)
            for v in values:
                if v == '': continue
                tmp = re.sub(r'\.', '-', v)
                text = re.sub(fr'\b{v}\b', fr' {grplist[0]}-{tmp} ', text)

    ##  7. 'Cho/Cr, Cho/NAA 수치의 기준 값 전처리.
    #  Cho/Cr = 2.29 (< 2.39). Cho/NAA = 1.14 (< 1.73).
    matches = re.findall(r'(Cho\-Cr|Cho\-NAA)\-(\d+\-\d+)\s*\(?([<>])\s*(\d+(?:\.\d+))\)?\.?', text)
    if matches:
        for grplist in matches:
            text = re.sub(fr'(?<={grplist[0]}-{grplist[1]})\s*\(?\<\s*', r' less-than ', text)
            text = re.sub(fr'(?<={grplist[0]}-{grplist[1]})\s*\(?\>\s*', r' greater-than ', text)
            tmp2 = grplist[3].replace('.', '-')
            text = re.sub(fr'({grplist[0]}-{grplist[1]}\s*(greater|less)-than)\s*{grplist[3]}\)?\.?',
                           fr' \1 {grplist[0]}-{tmp2} ', text)

    ##  8. 'ADC' 수치 데이터 전처리.
    #  ADC(avg) values of 0.862
    text = re.sub(r'ADC.*value(?:s)?.*?(\d+(?:\.\d+)?)\s*\)?\.?', r' adc-\1 ', text)
    text = re.sub(r'(adc-\d+)\.(\d+)', r' \1-\2 ', text)
    text = re.sub(r'(adc-\d+-\d+)[ \-]+>+\s*(\d+(?:\.\d+)?)', r' \1 change adc-\2 ', text)
    text = re.sub(r'(adc-\d+)\.(\d+)', r' \1-\2 ', text)

    # 특수문자가 포함된 일반적인 ADC 수치 데이터 정형화.
    matches = re.finditer(r'\(?(ADC)[ ,=]*(\d+(?:\.\d+)?(?:[, ]+\d+(?:\.\d+)?)*)\)\.?', text)
    # Ftext = re.sub(r'\(?ADC\s*', r' ', Ftext)   # ADC 값 정형화 전 'ADC' 삭제

    # (ADC 0.652, 1.062)    --> 'adc-0-652', 'adc-1-062'
    if matches:
        for grplist in matches:
            grp_values = grplist.group(2)
            values = re.findall(r"\d+(?:\.\d+)?", grp_values)
            for v in values:
                if v == '': continue
                tmp = re.sub(r'\.', r'-', v)
                text = re.sub(fr'{v}', fr'adc-{tmp}', text)

    ##  9. Score 또는 검사 대상 수치 데이터 전처리
    #  ex) SIH score = 1/10, cistern 1/1, collection 0/1 등
    text = re.sub(r'(score|sinus|enhancement|collection|cistern(?:a)?|distance)\s*[ \=]*(\d+)\/(\d+)'
                   , r' \1 pos-\2 tot-\3 '
                   , text)  # 1/10   -->  'pos-1', 'tot-10'

    ##  11. aneurysm의 4차원 값 표현 구분
    # Length + Width + Height(Depth) + Neck
    matches = re.findall(
        r'(\d{1,2}(?:\.\d{1,2})?)\s*(x|X|\*)\s*(\d{1,2}(?:\.\d{1,2})?)\s*(x|X|\*)\s*(\d(?:\.\d{1,2})?)\s*(x|X|\*)\s*(\d(?:\.\d{1,2})?)\s*\(neck\)\s*(mm|cm)',
        text)
    if matches:
        for grplist in matches:
            if grplist[-1] == 'mm':
                # Length 정형화
                if '.' in grplist[0]:
                    Ltmp = str(int(round(float(grplist[0]))))
                    Lvalue = re.sub(r'\.', r'\\.', grplist[0])
                else:
                    Ltmp, Lvalue = grplist[0], grplist[0]

                # Width 정형화
                if '.' in grplist[2]:
                    Wtmp = str(int(round(float(grplist[2]))))
                    Wvalue = re.sub(r'\.', r'\\.', grplist[2])
                else:
                    Wtmp, Wvalue = grplist[2], grplist[2]

                # Height 정형화
                if '.' in grplist[4]:
                    Htmp = str(int(round(float(grplist[4]))))
                    Hvalue = re.sub(r'\.', r'\\.', grplist[4])
                else:
                    Htmp, Hvalue = grplist[4], grplist[4]

                # Neck 정형화
                if '.' in grplist[6]:
                    Ntmp = str(int(round(float(grplist[6]))))
                    Nvalue = re.sub(r'\.', r'\\.', grplist[6])
                else:
                    Ntmp, Nvalue = grplist[6], grplist[6]

                text = re.sub(
                    fr'{Lvalue}\s*(x|X|\*)(?=\s*{Wvalue}\s*(x|X|\*)\s*{Hvalue}\s*(x|X|\*)\s*{Nvalue}\s*\(neck\)\s*mm)'
                    , fr' Length-{Ltmp}mm '
                    , text)

                text = re.sub(
                    fr'(?<=Length-{Ltmp}mm)\s*{Wvalue}\s*(x|X|\*)(?=\s*{Hvalue}\s*(x|X|\*)\s*{Nvalue}\s*\(neck\)\s*mm)'
                    , fr' Width-{Wtmp}mm '
                    , text)

                text = re.sub(
                    fr'(?<=Length-{Ltmp}mm Width\-{Wtmp}mm)\s*{Hvalue}\s*(x|X|\*)(?=\s*{Nvalue}\s*\(neck\)\s*mm)'
                    , fr' Height-{Htmp}mm '
                    , text)

                text = re.sub(fr'(?<=Length-{Ltmp}mm Width-{Wtmp}mm Height-{Htmp}mm)\s*{Nvalue}\s*\(neck\)\s*mm'
                               , fr' Neck-{Ntmp}mm '
                               , text)

    return text


# 문장 토큰화를 통해 [CLS], [SEP] 토큰 추가하기.
def sent_tokenizing(DSet: pd.DataFrame):
    nltk.download('punkt')      # 구두점 분리가 학습된 모델
    nltk.download('stopwords')  # 불용어 사전

    sentences_list = []

    # Findings, Conclusion 데이터를 모두 사용하므로 하나의 구성요소로 만든다.
    for idx, Fs in enumerate(zip(DSet.Findings, DSet.Conclusion)):
        # Findings에서 검사 시행 여부에 대한 내용은 삭제.
        #Ftext = ' '.join(map(str, Fs[0].split('\n'))).strip()
        Ftext = Fs[0].replace('\r', ' ')
        #Ctext = ' '.join(map(str, Fs[1].split('\n'))).strip()
        Ctext = Fs[1].replace('\r', ' ')
        text = Ftext + ' ' + Ctext

        text = prp.lobe_preprocessing(text)

        sentences = sent_tokenize(text)  # text를 tokenizing한 문장.
        Bert_sentences = ""
        for s in sentences:
            Bert_sentences += s + " __SEP__ "

        text = Bert_sentences

        text = semi_preprocessing(text)
        text = prp.medical_words_preprocessing(text)
        text = prp.cardi_ordinal_preprocessing(text)
        text = prp.pos_neg_preprocessing(text)
        text = prp.demention_preprocessing(text)
        text = prp.unnecessary_preprocessing(text)

        text = '[CLS] ' + text
        text = re.sub(r'__SEP__', r'[SEP]', text)

        while True:
            new_text = re.sub(r'(?:\[(SEP|CLS)\])[ ]+\[SEP\]', r'[\1]', text)
            if new_text == text:
                break
            text = new_text

        # BERT에 적용할 Train Record 저장
        sentences_list.append(text)

    # print('---- BERT 형식의 문장 생성 예시 ----')
    # print(sentences_list[100])
    # print('--------------------------------')

    return sentences_list


def BERT_Tokenizing_Model(sentences: list):
    global max_token_size

    # 한글 호환되는 토크나이저로 한글 단어에 대한 토큰을 추출하고, 이를 다른 토크나이저의 vocab에 추가한다.
    kor_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    kor_tokenizer.add_tokens(['폐쇄', '찢', '대뇌', '소뇌', '두통', '수술',
                              '감소', '증가', '관찰'])     # [UNK]로 분류되는 단어 또는 의학 용어 추가.
    for s in sentences :
        kts = kor_tokenizer.tokenize(s)

        words = []
        current_word = ''
        for token in kts:
            if token.startswith('##'):
                current_word += token[2:]
            else:
                if current_word:
                    words.append(current_word)
                current_word = token
        if current_word:
            words.append(current_word)

        hangul_only = [token for token in words if re.fullmatch(r'[가-힣#]+', token)]
        new_tokens = list(set(hangul_only))
        tokens_to_add = [tok for tok in new_tokens if tok not in tokenizer_bert.get_vocab()]
        tokenizer_bert.add_tokens(tokens_to_add)

    # BERT Tokenizer 최대 길이 = 512
    MAX_LEN = 450
    tokenized_sentences = []

    # [CLS] [SEP] 또는 [SEP] [SEP]가 발생하는 경우를 제거.
    # 모든 텍스트를 소문자로 변환.
    for s in sentences:
        tokens = tokenizer_bert.tokenize(s)

        words = []
        current_word = ''
        for token in tokens:
            if token.startswith('##'):
                current_word += token[2:]
            else:
                if current_word:
                    words.append(current_word)
                current_word = token
        if current_word :
            words.append(current_word)

        tokens_to_add = [tok for tok in words if tok not in tokenizer_bert.get_vocab()]
        tokenizer_bert.add_tokens(tokens_to_add)

        tokens = [tok if tok in ['[CLS]', '[SEP]', '[UNK]'] else tok.lower()
             for tok in words
             if (tok in ['[CLS]', '[SEP]', '[UNK]']) or (tok.lower() not in stopwords)]

        # if '[UNK]' in tokens:
        #     print(tokens)
        #     print(s)
        #     print('#################################')
        tokenized_sentences.append(tokens)

    # 단어 토큰에 고유한 인덱스 번호를 부여하고, 패딩을 첨가해 시퀀스 생성.
    input_ids = [tokenizer_bert.convert_tokens_to_ids(x) for x in tokenized_sentences]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="pre", padding="pre")


    #유의미한 토큰을 계산하고 최댓값을 추출하기 위함 (445개).
    for lst in input_ids:
        cnt = 0
        for l in lst:
            if l == tokenizer_bert.pad_token_id : continue
            cnt += 1
        if max_token_size < cnt :
            max_token_size = cnt
            #print(tokenizer_bert.convert_ids_to_tokens(lst))
            #print(cnt)

    return input_ids



def training(model : BertForSequenceClassification,
             device : torch.device,
             train_dataloader : DataLoader):

    # 모델을 gpu에 담기.
    model.to(device)
    # 토크나이저 단어 사전에 사용자 추가된 것이 있으므로 개수 반영.
    model.resize_token_embeddings(len(tokenizer_bert))
    # 옵티마이저
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    # 모델 에폭수
    epochs = 3
    # 총 훈련 스탭 = 배치 반복 횟수 * 에폭수
    total_steps = len(train_dataloader) * epochs
    # 스케줄러 생성
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps= int(0.1 * total_steps),
                                                num_training_steps=total_steps) # 또는 0

    # 에폭수만큼 배치 학습 반복 (조기 종료 추가)
    for epoch in range(epochs):
        total_loss = 0          # 평균 손실값 계산용
        # 모델을 학습 모드로 두고 진행.
        model.train()

        for step, batch in train_dataloader:
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


        # # 검증 정확도 계산 #
        # cstat = validation(test_dataloader)
        # print(f'[Epoch {epoch + 1}] Validation Accuracy = {cstat:.4f}')

        # # === Early stopping 조건 확인 ===
        # if cstat > best_val_acc:
        #     best_val_acc = cstat
        #     early_stop_counter = 0
        #     model.save_pretrained(model_save_path)
        #     print("✅ 모델 성능 향상 - 저장 완료")
        # else:
        #     early_stop_counter += 1
        #     print(f"⏸️ 개선 없음 - early_stop_counter = {early_stop_counter}/{patience}")
        #     if early_stop_counter >= patience:
        #         print("⛔ 조기 종료 조건 충족. 학습 종료.")
        #         break

    # 학습 완료한 모델 저장
    model.save_pretrained(model_save_path)
    tokenizer_bert.save_pretrained(model_save_path)



def validation(test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model2 = BertForSequenceClassification.from_pretrained('../../saved_bert_model_3')
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

    # 학습/테스트 DataFrame
    tdf = pd.DataFrame(kiumSet)
    vdf = pd.DataFrame(validSet)

    # mim = 2000
    # lim = 2050
    # tdf = tdf.iloc[mim:lim]


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


    #6. 학습에 필요한 gpu 활성화 및 모델 경로 설정
    device = prp.Checking_cuda()
    model_save_path = '../../saved_bert_model_3'

    #7. BERT 학습 모델.
    # 먼저 구성 객체 설정
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
    #training(model, device, train_dataloader)

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

'''