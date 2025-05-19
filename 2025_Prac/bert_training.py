'''
BERT tokenizer로 토큰화한 결과를 추가 처리 없이 직접 학습에 사용할 경우 정확도를 측정한다.
'''
import pandas as pd
import numpy as np
import nltk
import re
from nltk.tokenize import sent_tokenize
import Preprocessing as prp
from transformers import BertTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

        sentences = sent_tokenize(text)  # text를 tokenizing한 문장.
        Bert_sentences = ""
        for s in sentences:
            Bert_sentences += s + " __SEP__ "

        text = Bert_sentences

        text = prp.medical_words_preprocessing(text)
        text = prp.lobe_preprocessing(text)
        # text = prp.cardi_ordinal_preprocessing(text)
        # text = prp.pos_neg_preprocessing(text)
        text = prp.demention_preprocessing(text)
        text = semi_preprocessing(text)
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
    for s in sentences :
        kts = kor_tokenizer.tokenize(s)
        hangul_only = [token for token in kts if re.fullmatch(r'[가-힣]+', token)]
        new_tokens = list(set(hangul_only))
        tokens_to_add = [tok for tok in new_tokens if tok not in tokenizer_bert.get_vocab()]
        tokenizer_bert.add_tokens(tokens_to_add)

    # BERT Tokenizer 최대 길이 = 512
    MAX_LEN = 350
    tokenized_sentences = []
    for s in sentences:
        t = tokenizer_bert.tokenize(s)
        t = [tok if tok in ['[CLS]', '[SEP]'] else tok.lower()
             for tok in t
             if (tok in ['[CLS]', '[SEP]']) or (tok.lower() not in stopwords)]

        # MAX_LEN = max(MAX_LEN, len(t))
        tokenized_sentences.append(t[:MAX_LEN])

    # 단어 토큰에 고유한 인덱스 번호를 부여하고, 패딩을 첨가해 시퀀스 생성.
    input_ids = [tokenizer_bert.convert_tokens_to_ids(x) for x in tokenized_sentences]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="pre", padding="pre")

    # # 유의미한 토큰을 계산하고 최댓값을 추출하기 위함.
    # for lst in input_ids:
    #     cnt = 0
    #     for l in lst:
    #         if l == tokenizer_bert.pad_token_id :   continue
    #         cnt += 1
    #     if max_token_size < cnt :
    #         max_token_size = cnt
    #         print(tokenizer_bert.convert_ids_to_tokens(lst))
    #         print(cnt)

    return input_ids



if __name__ == '__main__':
    kiumSet = pd.read_csv(r'.\TrainCopySet.csv')
    #validSet = pd.read_csv(r'.\ValidationSet.csv')

    # 학습/테스트 DataFrame
    tdf = pd.DataFrame(kiumSet)
    #vdf = pd.DataFrame(validSet)

    mim = 180
    lim = 210
    tdf = tdf.iloc[mim:lim]


    # tokenizer ( PubMed 초록(abstract)만을 사용하여 처음부터 사전학습한 모델 )
    tokenizer_bert = BertTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')

    #1. 결측값 처리
    prp.empty_to_missing(tdf)
    #prp.empty_to_missing(vdf)


    #2. Findings + Conclusion 후, 문장 토큰화로 [CLS], [SEP] 토큰 추가하기.
    train_sentences = sent_tokenizing(tdf)
    #test_sentences = sent_tokenizing(vdf)

    """
    [CLS] Clinical information : Lung cancer. [SEP] Axial T1WI, sagittal T1WI, axial T2WI, axial FLAIR, axial T2* GRE image 획득하였으며 조영증강을 시행함. [SEP] Decreased size of heterogeneously enhancing large mass in left basal ganglia and frontal lobe(2.8x2.7cm -> 2.3x2.2cm). [SEP] -- With hemorrhagic transformation. [SEP] DDx. [SEP] 1) metastasis
           2) malignancy such as GBM [SEP]
    """

    #3. 토큰이 추가된 문장을 단어 토큰으로 생성 --> 단어 Sequence 생성.
    train_inputs = BERT_Tokenizing_Model(train_sentences)
    #test_inputs = BERT_Tokenizing_Model(test_sentences)

    #print(max_token_size) # 338

    #4. 정답지
    train_labels = tdf['AcuteInfarction'].values
    #test_labels = vdf['AcuteInfarction'].values


    for idx in range(0,30):
        print(tdf.iloc[idx].Findings + ' ' + tdf.iloc[idx].Conclusion)
        print()
        print(train_sentences[idx])
        print()
        print(tokenizer_bert.convert_ids_to_tokens(train_inputs[idx]))
        print('####################################################################################')
