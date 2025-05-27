"""
- TrainCopySet.csv : 학습용 원본 데이터 Set
- ValidationSet.csv : 검증(테스트) 원본 데이터 Set
- tensorflow 사용 시 Python3.9 버전에서는 2.11.0 Version을 사용한다  -->  Module Import 오류가 발생한다.
- python 3.9 이상의 버전에서 pytorch 설치 명령 링크 : https://pytorch.org/get-started/locally/

- 모델 학습을 위해서 'bertTraining.py' 사용.
- 모델 검증을 위해서 'bertValidating.py' 사용.
"""
import pandas as pd                         # DataFrame
import re                                   # Regular Expression
import torch, gc
import numpy as np
from transformers import BertTokenizer
import nltk                                 # Word Tokenization
from nltk.tokenize import sent_tokenize     # 문장 자연어 토큰화
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertConfig, AutoTokenizer
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score, classification_report

# Resource File Download
nltk.download('punkt')      # 구두점 분리가 학습된 모델 (분리 규칙 모델)
nltk.download('stopwords')  # 불용어 사전
#nltk.download('punkt_tab')

# 학습 모델 저장/불러오기 경로 설정
model_save_path = '../../saved_bert_model_6'


## 토큰화 사전에 없는 용어(UNK) 추가
#  이후에도 Token Embedding 작업에 사용.
#  bert-base-multilingual-cased : 다국어 지원 토크나이저
#  microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract : 의학 소견/논문 바탕의 pretrained model
vocab = ['찢어지는', '촤측', '없', '폐쇄', '상', '은', '과거', '않다', '없다', '보이다', '있다',
         'unremarkable', 'gammaknife', 'GKRS', 'corona']
tokenizer_bert = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')
#tokenizer_bert = BertTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
tokenizer_bert.add_tokens(vocab)
#tokenizer_bert = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

## 불용어 토큰 제거 Tuple.
# '-', '_'는 의미 구분 목적으로 사용할 것이므로 제외하지 않음.
stopwords = ('and', 'at', 'a', 'an', 'as', 'are', 'b', 'in', 'to', 'the', 'of', 'or',
             'm', 's',
             '가', '과', '그', '그리고', '는', '년', '등의', '또는', '로', '및', '볼', '분', '수', '서',
             '에서', '이에', '이', '인', '의', '외', '와', '을', '인해', '에는', '에', '에도',
             '중', '지', '현', '함', '~', '"',
             '.', ',', ':', '(', ')', '→', '[', ']', '/', '*', '=', '+', "'", '&', '#', '?', ';')


## DataFrame 'info' 출력.
def show_info(df: pd.DataFrame) -> None:
    """
    Display information about the DataFrame.
    :param df: The DataFrame.
    :return: None. print 'info()'
    """
    print('----------------------------------------\n' \
          '-------@@@@ 원본 데이터 셋 정보 @@@@-------\n' \
          '----------------------------------------')
    df.info()
    print('-----------------------------------------------------------')


## 결측값을 빈 문자열('')로 변환 처리.
#  Findings에는 1376개의 NaN(결측치) 데이터 존재.
#  Conclusion에는 34개의 NaN(결측치) 데이터 존재.
def empty_to_missing(df : pd.DataFrame) -> None:
    """
    Check the number of missing values in the 'Findings' and 'Conclusion' columns.
    Replace all missing values with empty strings.
    :param df: The DataFrame
    :return: None
    """
    print(f"Findings 결측값 = {df['Findings'].isnull().sum()}")
    print(f"Conclusion 결측값 = {df['Conclusion'].isnull().sum()}")

    # 모든 결측값에 'unremarkable' 의미의 단어 추가.
    df.loc[:, 'Findings'] = df['Findings'].fillna('finding unremarkable')
    df.loc[:, 'Conclusion'] = df['Conclusion'].fillna('conclusion unremarkable')

    print("@@@@ 결측값(NaN)을 빈 문자열('') 처리한다. @@@@\n -- 처리 결과 -- ")
    # 결측치 처리 결과
    print(f"Findings 결측값 처리 후 = {df['Findings'].isnull().sum()}")
    print(f"Conclusion 결측값 처리 후 = {df['Conclusion'].isnull().sum()}")
    print('-----------------------------------------------------------')


## 문장 토큰화 후 [SEP] 토큰 배치를 위한 '__SEP__' 추가.
# 문장 토큰화 후에는 불필요한 문장 삭제 및 전처리 작업 진행.
def sent_tokenizing(df : pd.DataFrame):
    """
    Adding '__SEP__' to separate sentences.
    :param df: The DataFrame
    :return: None
    """
    #결합 전, Conclusion에서 '이상 소견 없음', '특이 소견 없음' 등의 표현을 전처리.
    df.loc[:, 'Conclusion'] = df['Conclusion'].str.replace(
        r'(이상\s*소견\s*없음|특이\s*소견\s*없음)',
        'conclusion unremarkable',
        regex=True
    )
    # Findings + Conclusion
    df['context'] = df.Findings + ' __SEP__ ' + df.Conclusion

    # 2번 이상의 띄어쓰기(또는 줄넘김 등)를 1개로 처리.
    df.loc[:, 'context'] = df['context'].str.replace(r'\s+', ' ', regex=True).str.strip()

    # sent_tokenize()는 온점(.)을 기준으로 문장을 구분한다.
    # '.'을 포함한 약어는 미리 처리해야 무의미한 문장 구분을 없앨 수 있다.
    # ex) e.g, Rt. 등
    df.loc[:, 'context'] = (
        df['context']
        .str.replace(r'\b[rR][tT](\.|\s|$)', r' right ', regex=True)
        .str.replace(r'\b[lL][tT](\.|\s|$)', r' left ', regex=True)
        .str.replace(r'[eE]\.[gG]', r'example', regex=True)
    )

    index_list = df.index.tolist()
    for idx in index_list:
        sentences = sent_tokenize(df['context'].loc[idx])

        # sentences는 문장들의 list이므로 1개의 문자열로 다시 합친다.
        # 이때, 문장 구분 토큰인 '[SEP]'를 적용하기 위해 '__SEP__'로 임시 적용.
        text = ' __SEP__ '.join(sentences)
        df.loc[idx, 'context'] = text

        # print(sentences)
        # print('-----------------------------------------------------------------')
        # print(text)
        # print('###########################################################################')


## 대뇌의 4개(Parietal, Temporal, Occipital, Frontal)의 엽(Lobe) 분류 용어에 대한 정형화.
def lobe_preprocessing(df : pd.DataFrame):
    """
    Convert the names of the four cerebral lobes into a single token format.
    @Frontal lobe = 전두엽(운동, 판단, 언어)
    @Parietal lobe = 두정엽(공간 지각, 감각 정보)
    @Temporal lobe = 측두엽(청각, 기억)
    @Occipital lobe = 후두엽(시각 처리)
    :param text: Medical impression string data (Findings, Conclusion).
    :return: None
    """

    # 오타 정정
    df.loc[:, 'context'] = (
        df['context']
        .str.replace(   # cerebellum 오타 정정.
            r'cerebelli\.'
            , r'cerebellum'
            , regex=True
        )
        .str.replace(   # right 오타 = ight
            r'\bight'
            , r'right'
            , regex=True
        )
    )

    df.loc[:, 'context'] = (
        df['context']
        .str.replace(  #  both, bilateral 'parietal' + 'temporal' + 'occipital'
            r'[Bb]oth\s*[PTO]\-[PTO]\-[PTO]\s*(lobe(s)?)?|'
            r'(at )?(the )?([Bb]ilateral|[Bb]oth|양측) (temporal|parietal|occipital)[ &,-]*(temporal|parietal|occipital)(\,|\s|\&|and|\-)*(temporal|parietal|occipital)\s*(lobe(s)?|(\.|\,|and|\s)*(?=left|right))|'
            r'(at )?(the )?([Bb]ilateral|[bB]oth|양측) (parieto|temporo|occipito)[ \-,]*(parieto|temporo|occipito)[ \-,]*(occipital|temporal|parietal)\s*((lobe|area)s?|(\.|\,|and|\s)*((?=left|right|[lL]t|[rR]t)|(?!=fronto)))'
            , ' right-temporal-lobe right-parietal-lobe right-occipital-lobe left-temporal-lobe left-parietal-lobe left-occipital-lobe '
            , regex=True
        )
        .str.replace(  #  both, bilateral 'frontal' + 'parietal' + 'temporal'
            r'[Bb]oth\s*[PTF]\-[PTF]\-[PTF]\s*((lobe|area)(s)?)?|'
            r'(at )?the.*?([Bb]ilateral|[Bb]oth|양측).*frontal parietal temporal lobe(s)?|'
            r'(at )?(the )?([Bb]ilateral|[Bb]oth|양측) (temporal|parietal|frontal)[ &,]*(temporal|parietal|frontal)(\,|\s|\&|and|\-|및)*(temporal|parietal|frontal)\s*(lobe(s)?|(\.|\,|and|\s)*(?=left|right|[lL]t|[rR]t))|'
            r'(at )?(the)?([Bb]ilateral|[Bb]oth|양측) (fronto|parieto|temporo)[ ,\-]*(fronto|parieto|temporo)[ ,\-]*(frontal|parietal|temporal)\s*((lobe|area)s?|(\.|\,|and|\s)*((?=left|right|[lL]t|[rR]t)|(?!=occipit)))'
            , ' left-temporal-lobe left-parietal-lobe left-frontal-lobe right-temporal-lobe right-parietal-lobe right-frontal-lobe '
            , regex=True
        )
        .str.replace(  #  both, bilateral 'frontal' + 'parietal' + 'occipital'
            r'[Bb]oth\s*[POF]\-[POF]\-[POF]\s*(lobe(s)?)?|'
            r'(at )?(the )?([Bb]ilateral|[Bb]oth|양측) (occipital|parietal|frontal)[ &,]*(occipital|parietal|frontal)(\,|\s|\&|and|\-)*(occipital|parietal|frontal)\s*(lobe(s)?|(\.|\,|and|\s)*(?=left|right))|'
            r'(at )?(the )?([Bb]ilateral|[Bb]oth|양측) (fronto|parieto|occipito)[ ,\-&]*(fronto|parieto|occipito)[ ,\-&]*(occipital|parietal|frontal)\s*((lobe|area)s?|(\.|\,|and|\s)*((?=left|right|[lL]t|[rR]t)|(?!=tempor)))'
            , ' left-occipital-lobe left-parietal-lobe left-frontal-lobe right-occipital-lobe right-parietal-lobe right-frontal-lobe '
            , regex=True
        )
        .str.replace(  #  both, bilateral 'frontal' + 'temporal' + 'occipital'
            r'[Bb]oth\s*[TOF]\-[TOF]\-[TOF]\s*(lobe(s)?)?|'
            r'(at )?(the )?([Bb]ilateral|[Bb]oth|양측) (occipital|temporal|frontal)[ &,]*(occipital|temporal|frontal)(\,|\s|\&|and|\-)*(occipital|temporal|frontal)\s*(lobe(s)?|(\.|\,|and|\s)*(?=left|right))|'
            r'(at )?(the )?([Bb]ilateral|[Bb]oth|양측) (fronto|temporo|occipito)[ &,\-]*(fronto|temporo|occipito)[ &,\-]*(frontal|temporal|occipital)\s*((lobe|area)s?|(\.|\,|and|\s)*((?=left|right|[lL]t|[rR]t)|(?!=pariet)))'
            , ' left-occipital-lobe left-temporal-lobe left-frontal-lobe right-occipital-lobe right-temporal-lobe right-frontal-lobe '
            , regex=True
        )
        .str.replace(  #  both 'parietal' + 'temporal'
            r'[Bb]oth\s*[PT]\-[PT]\s*(lobe(s)?)?|both parieto-temporo-parietal lobe(s)?|'
            r'(at )?(the )?([Bb]ilateral|[Bb]oth|양측) (temporal|parietal)(\.|\,|and|\s|\&)*(temporal|parietal)\s*((area|lobe)s?|(\.|\,|and|\s)*((?=left|right)|(?!=occipit|front)))|'
            r'(at )?(the )?([Bb]ilateral|[bB]oth|양측) (parieto|temporo)[\- ,]*(parietal|temporal)\s*(lobe(s)?|(\.|\,|and|\s)*((?=left|right)|(?!=occipit|front)))'
            , ' right-temporal-lobe right-parietal-lobe left-temporal-lobe left-parietal-lobe '
            , regex=True
        )
        .str.replace(  #  both 'parietal' + 'occipital'
            r'[Bb]oth\s*[PO]\-[PO]\s*(lobe(s)?)?|'
            r'(at )?(the )?([Bb]ilateral|[Bb]oth|양측) (occipital|parietal|parieto|occipito)(\.|\,|and|\s|\&|\-)*(occipital|parietal)\s*((area|lobe)s?|(\.|\,|and|\s)*((?=left|right)|(?!=tempor|front)))'
            , ' right-occipital-lobe right-parietal-lobe left-occipital-lobe left-parietal-lobe '
            , regex=True
        )
        .str.replace(  #  both 'parietal' + 'frontal'
            r'([Bb]oth|the [Bb]ilateral)\s*[PF]\-[PF]\s*(lobe(s)?)?|'
            r'(at )?([Tt]he )?([Bb]ilateral|[Bb]oth|양측) (frontal|parietal|fronto|parieto)(\.|\,|and|\s|\&|\-|lobe)*(frontal|parietal)\s*((area|lobe)s?|(\.|\,|and|\s)*((?=left|right)|(?!=tempor|occipit)))'
            , ' right-parietal-lobe right-frontal-lobe left-parietal-lobe left-frontal-lobe '
            , regex=True
        )
        .str.replace(  #  both 'temporal' + 'occipital'
            r'[Bb]oth\s*[TO]\-[TO]\s*(lobe(s)?)?|'
            r'(at )?(the )?([Bb]ilateral|[Bb]oth|양측) (tempor|occipit?)(al|o)?(\.|\,|and|\s|\&|\-)*(temporal|occipital)\s*((area|lobe)s?|(\.|\,|and|\s)*((?=left|right)|(?!=pariet|front)))'
            , ' right-temporal-lobe right-occipital-lobe left-temporal-lobe left-occipital-lobe '
            , regex=True
        )
        .str.replace(  #  both 'temporal' + 'frontal'
            r'[Bb]oth\s*[TF]\-[TF]\s*(lobe(s)?)?|'
            r'(at )?([Tt]he )?([Bb]ilateral|[Bb]oth|양측) (frontal|fronto|temporal|temporo)(\.|\,|and|\s|\-|\&)*(frontal|temporal)\s*((area|lobe)s?|(\.|\,|and|\s)*((?=[Ll]eft|[Rr]ight)|(?!=pariet|occipit)))'
            , ' right-temporal-lobe right-frontal-lobe left-temporal-lobe left-frontal-lobe '
            , regex=True
        )
        .str.replace(  #  both 'occipital' + 'frontal'
            r'[Bb]oth\s*[OF]\-[OF]\s*(lobe(s)?)?|'
            r'(at )?([Tt]he )?([Bb]ilateral|[Bb]oth|양측) (frontal|fronto|occipital|occipito)(\.|\,|and|\s|\-|\&)*(frontal|occipital)\s*((area|lobe)s?|(\.|\,|and|\s)*((?=[Ll]eft|[Rr]ight)|(?!=pariet|tempor)))|'
            r'(at )?([Tt]he )?([Bb]ilateral|[Bb]oth|양측) (front|occipit)(al|o)( lobes?)?[ ,]*(front|occipit)(al|o)\s*lobes?'
            , ' right-occipital-lobe right-frontal-lobe left-occipital-lobe left-frontal-lobe '
            , regex=True
        )
        .str.replace(  #  both 'parietal'
            r'[Bb]oth P[ ,)]lobe(s)?|'
            r'(at )?(the )?([Bb]ilateral|[Bb]oth|양측)\s*par(i)?etal\s*((lobe|area)s?|(and|\,|\s|\&)*(?!=(front|occipit|tempor)))'
            , ' right-parietal-lobe left-parietal-lobe '
            , regex=True
        )
        .str.replace(  #  both 'temporal'
            r'[Bb]oth T[ ,)]lobe(s)?|'
            r'(at )?(the )?([Bb]ilater(r)?al|[Bb]oth|양측)\s*temporal\s*((lobe|area)s?|(and|\,|\s|\&)*(?!=(front|occipit|pariet)))'
            , ' right-temporal-lobe left-temporal-lobe '
            , regex=True
        )
        .str.replace(  #  both 'occipital'
            r'[Bb]oth O[ ,)]lobe(s)?|'
            r'(at )?(the )?([Bb]ilateral|[Bb]oth|양측)\s*occ(i)?p(i)?tal\s*((lobe|area)s?|(and|\,|\s|\&)*(?!=(front|tempor|pariet)))'
            , ' right-occipital-lobe left-occipital-lobe '
            , regex=True
        )
        .str.replace(  #  both 'frontal'
            r'[Bb]oth F[ ,)]lobe(s)?|'
            r'(at )?(the )?([Bb]ilateral|[Bb]oth|양측)\s*frontal\s*((lobe|area)s?|(and|\,|\s|\&)*(?!=(occipit|tempor|pariet)))'
            , ' right-frontal-lobe left-frontal-lobe '
            , regex=True
        )
        .str.replace(  #  both 'cerebellum' + 'cerebrum'
            r'(at )?(the )?([Bb]ilateral|[Bb]oth|양측)\s*(cerebral|cerebellar)( |and|\&|\-)*(cerebral|cerebellar)'
            , ' right-cerebellum right-cerebrum left-cerebellum left-cerebrum '
            , regex=True
        )
        .str.replace(  #  both 'cerebral' or 'cerebrum'
            r'(at )?(the )?([Bb]oth|[Bb]ilateral|bialteral|양측)\s*([Cc]erebral|[Cc]erebrum)'
            , ' right-cerebrum left-cerebrum '
            , regex=True
        )
        .str.replace(  #  both 'cerebellum'
            r'(at )?(the )?([Bb]oth|[Bb]ilateral|bialteral|양측)\s*[Cc]erebellum'
            , ' right-cerebellum left-cerebellum '
            , regex=True
        )
        .str.replace(  #  right 4개 항목
            r'(at )?(the )?([rR]ight|[rR]t\.?) (frontal|parietal|temporal|occipital)[ ,]*(frontal|parietal|temporal|occipital)[ ,]*(frontal|parietal|temporal|occipital)( |\,|and)*(frontal|parietal|temporal|occipital)[ ,]*(lobe|area)s?'
            , ' right-temporal-lobe right-parietal-lobe right-occipital-lobe right-frontal-lobe '
            , regex=True
        )
        .str.replace(  #  right 'parietal' + 'temporal' + 'occipital'
            r'(at )?(the )?([rR]ight|[rR]t)\.?\s*[PTO]\-[PTO]\-[PTO]\s*((lobe|area)(s)?)?|'
            r'(at )?(the )?([Rr]ight|[rR]t\.?) (pariet|tempor|occipit)(al|o)[, &\-]*(pariet|tempor|occipit)(al|o)(\,|\s|\&|and|\-)*(pariet|tempor|occ(i)?pit)(al|o)(\,|\s|\&|and|\-)*((lobe|area)s?\.?|(?=left|both)|(?!=front))|'
            r'(at )?(the )?([Rr]ight|[rR]t\.?) [PTO]+\(((pariet|tempor|occipit)(al|o)|\s|\-)+\)\s*lobes?\.?'
            , ' right-temporal-lobe right-parietal-lobe right-occipital-lobe '
            , regex=True
        )
        .str.replace(  #  right 'parietal' + 'temporal' + 'frontal'
            r'(at )?(the )?([rR]ight|[rR]t)\.?\s*[PTF]\-[PTF]\-[PTF]\s*((lobe|area)(s)?)?|'
            r'(at )?(the )?([Rr]ight|[rR]t\.?) (pariet|tempor|fron(t)?)(al|o)[, &\-]*(pariet|tempor|front)(al|o)(\,|\s|\&|and|\-)*(pariet|tempor|front)(al|o)(\,|\s|\&|and|\-)*((lobe|area)s?\.?|(?=left|both)|(?!=occipit))'
            , ' right-temporal-lobe right-parietal-lobe right-frontal-lobe '
            , regex=True
        )
        .str.replace(  #  right 'parietal' + 'occipital' + 'frontal'
            r'(at )?(the )?([rR]ight|[rR]t)\.?\s*[POF]\-[POF]\-[POF]\s*((lobe|area)(s)?)?|'
            r'(at )?(the )?([Rr]ight|[rR]t\.?) (pariet|occipit|front)(al|o)[, &\-]*(pariet|occipit|front)(al|o)(\,|\s|\&|and|\-)*(pariet|occ(i)?pit|front)(al|o)(\,|\s|\&|and|\-)*((lobe|area)s?\.?|(?=left|both)|(?!=tempor))'
            , ' right-occipital-lobe right-parietal-lobe right-frontal-lobe '
            , regex=True
        )
        .str.replace(  #  right 'temporal' + 'occipital' + 'frontal'
            r'(at )?(the )?([rR]ight|[rR]t)\.?\s*[TOF]\-[TOF]\-[TOF]\s*((lobe|area)(s)?)?|'
            r'(at )?(the )?([Rr]ight|[rR]t\.?) (tempor|occipit|front)(al|o)[, &\-]*(tempor|occipit|front)(al|o)(\,|\s|\&|and|\-)*(tempor|occ(i)?pit|front)(al|o)(\,|\s|\&|and|\-)*((lobe|area)s?\.?|(?=left|both)|(?!=pariet))'
            , ' right-occipital-lobe right-temporal-lobe right-frontal-lobe '
            , regex=True
        )
        .str.replace(  #  right 'parietal' + 'temporal'
            r'(at )?(the )?([Rr](i)?ght|[rR]t)\.?\s*[PT]\-[PT]\s*((lobe|area)(s)?)?|'
            r'(at )?(the )?([Rr]ight|[rR]t\.?) (pariet|tempor)(al|o)(\,|\s|\&|and|\-|lobe)*(pariet|tempor)(al|o)(\,|\s|\&|and|\-)*((lobe|area)s?\.?|(?=left|both)|(?!=occipit|front))'
            , ' right-temporal-lobe right-parietal-lobe '
            , regex=True
        )
        .str.replace(  #  right 'parietal' + 'occipital'
            r'(at )?(the )?(right|[rR]t)\.?\s*[PO]\-[PO]\s*((lobe|area)(s)?)?|'
            r'(at )?(the )?([Rr]ight|[rR]t\.?) (pariet|occipit)(al|o)(\,|\s|\&|and|\-)*(pariet|occipit)(al|o)(\,|\s|\&|and|\-)*((lobe|area)s?\.?|(?=left|both)|(?!=tempor|front))'
            , ' right-occipital-lobe right-parietal-lobe '
            , regex=True
        )
        .str.replace(  #  right 'parietal' + 'frontal'
            r'(at )?(the )?(right|[rR]t)\.?\s*[FP](\-|\, )[FP]\s*((lobe|area)(s)?)?|'
            r'(at )?(the )?([rR]ight|[rR]t\.?) (front|pariet)(al|o)(\,|\s|\&|and|\-|lobe)*(front|pariet)(al|o)(\,|\s|\&|and|\-)*((lobe|area)s?\.?|(?=left|both)|(?!=tempor|occipit))'
            , ' right-frontal-lobe right-parietal-lobe '
            , regex=True
        )
        .str.replace(  #  right 'temporal' + 'occipital'
            r'(at )?(the )?(right|[rR]t)\.?\s*[TO]\-[TO]\s*((lobe|area)(s)?)?|'
            r'(at )?(the )?([rR]ight|[rR]t\.?) (tempor|occipit)(al|o)(\,|\s|\&|and|\-)*(tempor|occipit)(al|o)(\,|\s|\&|and|\-)*((lobe|area)s?\.?|(?=left|both)|(?!=front|pariet))'
            , ' right-temporal-lobe right-occipital-lobe '
            , regex=True
        )
        .str.replace(  #  right 'temporal' + 'frontal'
            r'(at )?(the )?(right|[rR]t)\.?\s*[FT]\-[FT]\s*((lobe|area)(s)?)?|'
            r'(at )?(the )?([rR]ight|[rR]t\.?) (tempor|f(ro|or)nt)(al|o)(\,|\s|\&|and|\-)*(tempor|front|temopr)(al|o)(\,|\s|\&|and|\-)*((lobe|area)s?\.?|(?=left|both)|(?!=occipit|pariet))'
            , ' right-temporal-lobe right-frontal-lobe '
            , regex=True
        )
        .str.replace(  #  right 'occipital' + 'frontal'
            r'(at )?(the )?(right|[rR]t)\.?\s*[OF]\-[OF]\s*((lobe|area)(s)?)?|'
            r'(at )?(the )?([Rr]ight|[rR]t\.?) (occipit|front)(al|o)(\,|\s|\&|and|\-)*(occipit|front)(al|o)(\,|\s|\&|and|\-)*((lobe|area)s?\.?|(?=left|both)|(?!=tempor|pariet))'
            , ' right-occipital-lobe right-frontal-lobe '
            , regex=True
        )
        .str.replace(  #  right 'parietal'
            r'(at )?(the )?([rR][Tt]\.?|[Rr]ight) P[ ,).]|'
            r'(at )?(the )?([Rr]ight|[rR][tT]\.?) (pari(e)?t|paret)(al?|o)(\,|\s|\&|and|\-)*((lobe|area)(s|에)?\.?|(?=left|both)|(?!=(occipit|front|tempor)))|'
            r'우측 두정엽'
            , ' right-parietal-lobe '
            , regex=True
        )
        .str.replace(  #  right 'temporal'
            r'([rR][Tt]\.?|[Rr]ight) T[ ,).]|'
            r'(at )?(the )?([Rr]ight|[rR][tT]\.?) tempor(al|o)(\,|\s|\&|and|\-)*((lobe|area)(s|에)?[.-]*|(?=left|both)|(?!=(occipit|front|pariet)))|'
            r'우측 측두엽'
            , ' right-temporal-lobe '
            , regex=True
        )
        .str.replace(  #  right 'occipital'
            r'([rR][Tt]\.?|[Rr]ight) O[ ,).]|'
            r'(at )?(the )?([Rr]ight|[rR][tT]\.?) (occ(i)?pit|occipti)(al|o)(\,|\s|\&|and|\-)*((lobe|area)(s|에)?\.?|(?=left|both)|(?!=(pariet|front|tempor)))|'
            r'우측 후두엽'
            , ' right-occipital-lobe '
            , regex=True
        )
        .str.replace(  #  right 'frontal'
            r'([rR][Tt]\.?|[Rr]ight) F[ ,).]|'
            r'(at )?(the )?([Rr]ig(ht|th)|[rR][tT]\.?) front(al|o)(\,|\s|\&|and|\-)*((lobe|area)(s|에)?\.?|(?=[Ll]eft|[lL]t|both)|(?!=(pariet|occipit|tempor)))|'
            r'우측 전두엽'
            , ' right-frontal-lobe '
            , regex=True
        )
        .str.replace(  #  right 'cerebellum'
            r'(at )?(the )?([rR][Tt]\.?|[Rr]ight) ([cC][Bb][lL][lL]|cerebellum|cerebellar)\.?'
            , ' right-cerebellum '
            , regex=True
        )
        .str.replace(  #  right 'cerebrum'
            r'(at )?(the )?([rR][Tt]\.?|[Rr]ight) (cer(e)?bral|cerebrum)\.?'
            , ' right-cerebrum '
            , regex=True
        )
        .str.replace(  #  left 4개 항목
            r'(at )?(the )?([lL]eft|[lL]t\.?) (front|pariet|tempor|occipit)(al|o)[ ,\-]*(front|pariet|tempor|occipit)(al|o)[ ,\-]*(front|pariet|tempor|occipit)(al|o)( |\,|and|\-)*(front|pariet|tempor|occipit)(al|o)[ ,]*(lobe|area)s?'
            , ' left-temporal-lobe left-parietal-lobe left-occipital-lobe left-frontal-lobe left-cerebellum '
            , regex=True
        )
        .str.replace(  #  left 'parietal' + 'temporal' + 'ocipital'
            r'([Ll]eft|[lL]t)\.?\s*[PTO]\-[PTO]\-[PTO]\s*(lobe(s)?)?|'
            r'(at )?(the )?([lL]eft|[lL]t\.?) (pariet|tempor|occipit)(al|o)[, &\-]*(pariet|tempor|occipit)(al|o)(\,|\s|\&|and|\-)*(pariet|tempor|occipit)(al|o)(\,|\s|\&|and|\-)*((lobe|area)s?\.?|(?=[Rr]ight|both)|(?!=front))'
            , ' left-temporal-lobe left-parietal-lobe left-occipital-lobe '
            , regex=True
        )
        .str.replace(  #  left 'parietal' + 'temporal' + 'frontal'
            r'([Ll]eft|[lL]t)\.?\s*[PTF]\-[PTF]\-[PTF]\s*(lobe(s)?)?|'
            r'(at )?(the )?([lL]eft|[lL]t\.?) (pariet|tempor|front|fornt)(al|o)(\,|\s|\&|and|\-)*(pariet|tempor|front)(al|o)(\,|\s|\&|and|\-)*(pariet|tempor|front)(al|o)(\,|\s|\&|and|\-)*((lobe|area)s?\.?|(?=[Rr]ight|both)|(?!=occipit))'
            , ' left-temporal-lobe left-parietal-lobe left-frontal-lobe '
            , regex=True
        )
        .str.replace(  #  left 'parietal' + 'occipital' + 'frontal'
            r'([Ll]eft|[lL]t)\.?\s*[POF]\-[POF]\-[POF]\s*(lobe(s)?)?|'
            r'(at )?(the )?([lL]eft|[lL]t\.?) (pariet|occipit|front)(al|o)(\s*\(prefrontal gyrus\))?[, &\-]*(pariet|occipit|front)(al|o)(\,|\s|\&|and|\-)*(pariet|occ(i)?pit|front)(al|o)(\,|\s|\&|and|\-)*((lobe|area)s?\.?|(?=[Rr]ight|both)|(?!=tempor))'
            , ' left-occipital-lobe left-parietal-lobe left-frontal-lobe '
            , regex=True
        )
        .str.replace(  #  left 'temporal' + 'occipital' + 'frontal'
            r'([Ll]eft|[lL]t)\.?\s*[TOF]\-[TOF]\-[TOF]\s*(lobe(s)?)?|'
            r'(at )?(the )?([lL]eft|[lL]t\.?) (tempor|occipit|front)(al|o)[, &\-]*(tempor|occipit|front)(al|o)(\,|\s|\&|and|\-)*(tempor|occipit|front)(al|o)(\,|\s|\&|and|\-)*((lobe|area)s?\.?|(?=[Rr]ight|both)|(?!=pariet))'
            , ' left-temporal-lobe left-occipital-lobe left-frontal-lobe '
            , regex=True
        )
        .str.replace(  #  left 'parietal' + 'temporal'
            r'([Ll]eft|[lL]t)\.?\s*[PT]\-[PT]\s*((lobe|area)(s)?)?|'
            r'(at )?(the )?([Ll]eft|[lL]t\.?) (pariet|tempor)(al|o)( |and|\,|\&|\-|lobe)*(pariet|tempor)(al|o)(\,|\s|\&|and|\-)*((lobe|area)s?\.?|(?=[rR]ight|both)|(?!=front|occipit))'
            , ' left-temporal-lobe left-parietal-lobe '
            , regex=True
        )
        .str.replace(  #  left 'parietal' + 'occipital'
            r'([Ll]eft|[lL]t)\.?\s*[PO]\-[PO]\s*((lobe|area)(s)?)?|'
            r'(at )?(the )?([lL]eft|[lL]t\.?) (pari(et|te)|occipit)(al|o)( |\&|\,|and|\-|lobe)*(pariet|occipit)(al|o)(\,|\s|\&|and|\-)*((lobe|area)s?\.?|(?=[rR]ight|both)|(?!=front|tempor))'
            , ' left-occipital-lobe left-parietal-lobe '
            , regex=True
        )
        .str.replace(  #  left 'parietal' + 'frontal'
            r'([Ll]eft|[lL]t)\.?\s*[PF]\-[PF]\s*((lobe|area)(s)?)?|'
            r'the left.*?F, P lobes|'
            r'(at )?(the )?([lL]eft|[lL]t\.?) (front|pariet)(al|o)( |\&|\,|and|\-|lobe)*(front|pariet|paro)(al|o)(\,|\s|\&|and|\-)*((lobe|area)s?\.?|(?=[rR]ight|both)|(?!=occipit|tempor))'
            , ' left-frontal-lobe left-parietal-lobe '
            , regex=True
        )
        .str.replace(  #  left 'temporal' + 'occipital'
            r'([Ll]eft|[lL]t)\.?\s*[TO]\-[TO]\s*((lobe|area)(s)?)?|'
            r'(at )?(the )?([lL]eft|[lL]t\.?)\s*(occipit|tempor)(al|o)( |\&|\,|and|\-|lobe)*(occipit|tempor)(al|o)(\,|\s|\&|and|\-)*((lobe|area)s?\.?|(?=[rR]ight|both)|(?!=front|pariet))'
            , ' left-temporal-lobe left-occipital-lobe '
            , regex=True
        )
        .str.replace(  #  left 'temporal' + 'frontal'
            r'([Ll]eft|[lL]t)\.?\s*[FT]\-[FT]\s*((lobe|area)(s)?)?|'
            r'(at )?(the )?([lL]eft|[lL]t\.?) (front|tempor)(al|o)(\s|\&|\,|and|\-|lobe)*(front|tempor)(al|o)(\,|\s|\&|and|\-)*((lobe|area)s?\.?|(?=[rR]ight|both)|(?!=occipit|pariet))'
            , ' left-temporal-lobe left-frontal-lobe '
            , regex=True
        )
        .str.replace(  #  left 'occipital' + 'frontal'
            r'([Ll]eft|[lL]t)\.?\s*[OF]\-[OF]\s*(lobe(s)?)?|'
            r'(at )?(the )?([lL]eft|[lL]t\.?) (occipit|front)(al|o)(\s|\&|\,|and|\-|lobe)*(occipit|front)(al|o)(\,|\s|\&|and|\-)*((lobe|area)s?\.?|(?=[rR]ight|both)|(?!=pariet|tempor))'
            , ' left-occipital-lobe left-frontal-lobe '
            , regex=True
        )
        .str.replace(  #  left 'parietal'
            r'([lL][Tt]\.?|[Ll]eft) P[ ,)]|'
            r'(at )?(the )?([Ll]eft|[lL][Tt]\.?|elft) (p(a)?riet|parite)(al|o)(\,|\s|\&|and|\-)*((lobe|area)s?\.?|(?=[rR]ight|both)|(?!=(front|occipit|tempor)))|'
            r'좌측 두정엽'
            , ' left-parietal-lobe '
            , regex=True
        )
        .str.replace(  #  left 'temporal'
            r'([lL][Tt]\.?|[lL]eft) T[ ,)]|'
            r'(at )?(the )?([Ll]eft|[lL][Tt]\.?)\s*(tempo[tr]|tempror)(al|o)(\,|\s|\&|and|\-|\.)*((lobe|area)s?\.?|(?=[rR]ight|both)|(?!=(front|occipit|pariet)))|'
            r'좌측 측두엽'
            , ' left-temporal-lobe '
            , regex=True
        )
        .str.replace(  #  left 'occipital'
            r'([lL][Tt]\.?|[Ll]eft) O[ ,)]|'
            r'(at )?(the )?([lL]eft|[lL][Tt]\.?) (occ(i)?pit|occip(i)?ti)(al|o)(\,|\s|\&|and|\-)*((lobe|area)s?\.?|(?=[rR]ight|both)|(?!=(front|pariet|tempor)))|'
            r'좌측 후두엽'
            , ' left-occipital-lobe '
            , regex=True
        )
        .str.replace(  #  left 'frontal'
            r'([lL][Tt]\.?|[Ll]eft) F[ ,):]|'
            r'(at )?(the )?([Ll]e(f)?t|[lL][tT]\.?) front(al|o)(\,|\s|\&|and|\-)*((lobe|area|\,)s?\.?|(?=[rR]ight|both)|(?!=(occipit|pariet|tempor)))|'
            r'좌측 전두엽'
            , ' left-frontal-lobe '
            , regex=True
        )
        .str.replace(  # left 'cerebellum'
            r'(the )?([lL][Tt]\.?|[lL]eft) ([cC][Bb][lL][lL]|c(e)?rebellum|cerebellar)\.?'
            , ' left-cerebellum '
            , regex=True
        )
        .str.replace(  #  left 'cerebrum'
            r'(the )?([lL][Tt]\.?|[lL]eft) (cerebral|cerebrum)\.?'
            , ' left-cerebrum '
            , regex=True
        )
        #  기타 조합의 경우 별도 정형화 작업
        .str.replace(  #  left 'parietal' + 'temporal' + 'frontal' + 'cerebellum'
            r'left frontal parietal temporal lobes and cerebellum'
            , ' left-frontal-lobe left-parietal-lobe left-temporal-lobe left-cerebellum '
            , regex=True
        )
        .str.replace(  #  right 'temporal' + 'occipital' + 'cerebellum'
            r'right temporooccipital lobe and cerebellum'
            , ' right-occipital-lobe right-temporal-lobe right-cerebellum '
            , regex=True
        )
        .str.replace(  #  both 'parietal' + 'temporal' + 'cerebellum'
            r'both parietotemporal lobes cerebellum'
            , ' left-parietal-lobe left-temporal-lobe left-cerebellum right-parietal-lobe right-temporal-lobe right-cerebellum '
            , regex=True
        )
        .str.replace(  #  both 'parietal' + 'frontal' + 'cerebellum'
            r'both frontoparietal lobes cerebellum'
            , ' left-parietal-lobe left-frontal-lobe left-cerebellum right-parietal-lobe right-frontal-lobe right-cerebellum '
            , regex=True
        )
        .str.replace(  # 'orbitofrontal'
            r'([rR]ight|[lL]eft|[bB]oth)\s*orbitofrontal\s*(?:lobe|area)(s)?'
            , ' \1-orbitofrontal-lobe '
            , regex=True
        )
    )


# 기타 세부적인 표현에 대한 전처리 작업.
def semi_preprocessing(df : pd.DataFrame):
    tmp = df['context'].copy()      # 비교용 복삽본

    df.loc[:, 'context'] = (
        df['context']
        .str.replace(  # 'ms'를 의미하는 수치 데이터 전처리.
            r'\b(\d+)\s*(MS|ms)\b|TE\s+(\d+)'
            , lambda m: (
                f"ms_{m.group(1)}" if m.group(1) else           # \1: 숫자 (ms)
                f"te ms_{m.group(3)}" if m.group(3) else ""     # TE 144 등의 ms없는 표현
            )
            , regex=True
        )
        .str.replace(               # % = percent 값 전처리.
            r'(\d+)\s*\%'
            , r' percent_\1 '       # 10%  --> percent-10
            , regex=True
        )
    )

    for idx, text in df['context'].items():
        ## Cho/NAA 수치 데이터 전처리.
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
                    tmp = re.sub(r'\.', '_', v)
                    text = re.sub(fr'\b{v}\b', fr' {grplist[0]}_{tmp} ', text)


        ## 'Cho/Cr, Cho/NAA 수치의 기준 값 전처리.
        #  Cho/Cr = 2.29 (< 2.39). Cho/NAA = 1.14 (< 1.73).
        matches = re.findall(r'(Cho\-Cr|Cho\-NAA)\-(\d+\-\d+)\s*\(?([<>])\s*(\d+(?:\.\d+))\)?\.?', text)
        if matches:
            for grplist in matches:
                text = re.sub(fr'(?<={grplist[0]}_{grplist[1]})\s*\(?\<\s*', r' less_than ', text)
                text = re.sub(fr'(?<={grplist[0]}_{grplist[1]})\s*\(?\>\s*', r' greater_than ', text)
                tmp2 = grplist[3].replace('.', '_')
                text = re.sub(fr'({grplist[0]}_{grplist[1]}\s*(greater|less)_than)\s*{grplist[3]}\)?\.?',
                              fr' \1 {grplist[0]}_{tmp2} ', text)

        ## 'ADC' 수치 데이터 전처리.
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


        ## aneurysm의 4차원 값 표현 구분
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

        # 전처리된 text 저장(덮어쓰기)
        df.at[idx, 'context'] = text

    # Score 또는 검사 대상 수치 데이터 전처리
    # ex) SIH score = 1/10, cistern 1/1, collection 0/1 등
    df.loc[:, 'context'] = (
        df['context']
        .str.replace(  # 1/10   -->  'pos-1', 'tot-10'
            r'(score|sinus|enhancement|collection|cistern(?:a)?|distance)\s*[ \=]*(\d+)\/(\d+)'
            , r' \1 pos_\2 tot_\3 '
            , regex=True
        )
    )


    # 한국어 표현의 정형화
    df.loc[:, 'context'] = (
        df['context']
        .str.replace(
            r'않(고|다|으며|는다|음|았음|아)'
            , '않다'
            , regex=True
        )
        .str.replace(
            r'없(고|다|으며|음|었음|어|어보임)'
            , '없다'
            , regex=True
        )
        .str.replace(
            r'있(고|다|으며|음|어|었떤)'
            , '있다'
            , regex=True
        )
        .str.replace(
            r'보(임|이다|이며|인다|이는|이고|여)'
            , '보이다'
            , regex=True
        )
    )



## 소견문에 포함된 의학 용어(약어 및 기호를 포함한 단어) 정형화 함수.
#  Findings, Conclusion에 작성된 의학 용어 정형화.
def medical_words_preprocessing(df : pd.DataFrame) :
    """
        Medical terminology normalization function for clinical findings.
        :param text: DataFrame.
        :return: None.
    """
    df.loc[:, 'context'] = (
        df['context']
        .str.replace(
            r'(Axial[\- ]+(T1WI|DWI)|Intracranial[\- ]+TOF[\- ]+MRA|Axial[\- ]+T2\*|Neck[\- ]+MRA)[, ]*(sagittal[\- ]+T1WI)?[, ]*'
            r'(axial[\- ]+T2WI)?[, ]*(axial[\- ]+FLAIR)?[, ]*'
            r'(axial[\- ]+T2\*)?[, ]*(axial[\- ]+t2_star)?[, ]*(GRE image)?[, ]*(axial[\- ]+DWI)?[, ]*'
            r'(intracranial[\- ]+TOF[\- ]+MRA)?[, ]*(axial[\- ]+FLAIR)?[, ]*'
            r'(neck[\- ]+TOF[\- ]+MRA)?[, ]*.*조영증강[은을 ]+시행(함|하지 않았?음)\.?|'
            r'White matter tract evaluation.*diffusion tensor imaging 시행(함|하지 않았?음)\.?|'
            r'(MR brain venography|PC3D Brain MR venography).*시행(함|하지 않았?음)\.?'
            , ' '
            , regex=True
            , flags = re.IGNORECASE
        )
        .str.replace(   # No + 용어를 1개의 토큰으로 만듦.
            r'([nN]o\s)((?!at )\w+)(?:(?=\s|\.|\,|$))'
            , r' \1-\2 '
            , regex=True
        )
        .str.replace(   # stage 3,
            r'([sS]tage)\s*(\d+)[,. ]+'
            , r' \1_\2 '
            , regex=True
        )
        .str.replace(   # (CE)
            r'\(CE\)'
            , ' contrast-enhancement '
            , regex=True
        )
        .str.replace(   # (Non CE)
            r'\([nN]on CE\)'
            , ' Non-contrast-enhancement '
            , regex=True
        )
        .str.replace(   # /c = with
            r'\/c\s'
            , r' with '
            , regex=True
        )
        .str.replace(   # c spine  = c-spine
            r'\b[cC] [sS]pine'
            , r' c_spine '
            , regex=True
        )
        .str.replace(   # lung(IA)-NSCLC
            r'\(IA\)\-?'
            , ' stage_ia '
            , regex=True
        )
        .str.replace(   # anterior communicating artery
            r'[pP]\-[cC][Oo][Mm]\.?\s*a((\.|\))\)?|rtery)|'
            r'anterior communicating artery'
            , ' posterior-communicating-artery '
            , regex=True
        )
        .str.replace(   # posterior communicating, PCOM
            r'[pP]\-[cC][Oo][Mm](\.|\&)?|[pP][cC][oO][mM](\.|\&)?|'
            r'posterior communicating'
            , ' posterior-communicating '
            , regex=True
        )
        .str.replace(   # anterior communicating, ACOM
            r'[aA]\-[cC][oO][mM]\.?|[aA][cC][oO][mM]\.?|'
            r'anterior communicating'
            , ' anterior-communicating '
            , regex=True
        )
        .str.replace(   # large B-cell lymphoma, DLBCL
            r'[Ll]arge [bB]-cell lymphoma'
            , r'dlbcl'
            , regex=True
        )
        .str.replace(   # Non-small cell lung cancer.
            r'[nN]on[- ]*small cell lung cance(r)?'
            , r'nsclc'
            , regex=True
        )
        .str.replace(   # Von Hippel–Lindau disease.
            r'[vV]on [Hh]ippel[- ][lL]indau [Dd]isease'
            , r'vhl_disease'
            , regex=True
        )
        .str.replace(   # 2개 혼합 용어
            r'([eE]vans|[Cc]allosal)\s*([iI]ndex|[aA]ngle)'
            , r' \1-\2 '
            , regex=True
        )
        .str.replace(   # MM을 길이 단위 mm으로 혼용되지 않도록.
            r'\bMM[.,]'
            , r' multiple-myeloma '
            , regex=True
        )
        .str.replace(   # Intracranial TOF MRA
            r'[iI]ntracranial [tT][oO][fF] [mM][rR][aA]'
            , ' intracranial-tof-mra '
            , regex=True
        )
        .str.replace(   # Neck TOF MRA
            r'[Nn]eck [tT][oO][fF] [mM][rR][aA]'
            , 'neck-tof-mra'
            , regex=True
        )
        .str.replace(   # Neck MRA
            r'[Nn]eck [mM][rR][aA]'
            , 'neck-mra'
            , regex=True
        )
        .str.replace(   # 감마나이프, Gamma knife
            r'감마나이프|[gG]amma[ \-][kK]nife'
            , ' gammaknife '
            , regex=True
        )
        .str.replace(   # Clinical information, CI:, CI,  (삭제 목적)
            r'Clinical information\s*:|\*\s*CI\s?:|C\.?I[,: ;]+'
            , ''
            , regex=True
        )
        .str.replace(   # s/p, S/P
            r'[sS]\/[pP]'
            , ' status_post '
            , regex=True
        )
        .str.replace(   # r/o, R/O
            r'[rR][/][oO]'
            , ' rule_out '
            , regex=True
        )
        .str.replace(   # op. site
            r'[Oo]p\.\s*site[., (]|(at )?(the )?op site'
            , ' operative_site '
            , regex=True
        )
        .str.replace(   # h/o
            r'[hH]\/[oOpP]|history of'
            , r' history_of '
            , regex=True
        )
        .str.replace(   # f/u, f-u, f.u
            r'[Ff][./-][Uu]|follow up|follow\-up|'
            r'Fu (?=MR(I|A))'
            , ' follow_up '
            , regex=True
        )
        .str.replace(   # N/V = Nausea and Vomiting
            r'N\/V'
            , r'nausea vomiting'
            , regex=True
        )
        .str.replace(    # T2*
            r'[tT]2\*'
            , r' t2_star '
            , regex=True
        )
        .str.replace(   # T2/FLAIR
            r'[Tt]2[/\-][fF][lL][aA][iI][rR]'
            , r' t2_flair '
            , regex=True
        )
        .str.replace(   # t2,1 hyperintense
            r'\b([Tt][12]) hyperintens(e|ities|ity)'
            , r' \1_hyperintense '
            , regex=True
        )
        .str.replace(   # w/u, W/U.
            r'\b[wW][/-][uU]\.?\b'
            , r' work_up '
            , regex=True
        )
        .str.replace(    # jx.
            r'jx\.'
            , r' junction '
            , regex=True
        )
        .str.replace(   # inverted T 등
            r'[iI]nverted ([TVYU])'
            , r' inverted_\1 '
            , regex=True
        )
        .str.replace(   # CN V
            r'(CN|[cC]ranial [nN]erve)\s*([IV1-3]+)'
            , lambda m:
                ' cn cn_ophthalmic' if m.group(2) == 'V1' else
                ' cn cn_maxilary' if m.group(2) == 'V2' else
                ' cn cn_mandibular' if m.group(2) == 'V3' else r' cn_v '
            , regex=True
        )
        .str.replace(   # LC(B, CP:6A)   - Child-Pugh score
            r'(?<=LC)[ \(]*([ABC])[, ]*(?:CP)?[ :,]([0-9ABC]+)\)?'
            , r' lc_grade_\1_\2 '
            , regex=True
        )
        .str.replace(   # grade 2 등
            r'(?:[gG]rade|[gG]r\.)\s*(\d+|[iI]+)\s*\)?\.?'
            , r' grade_\1 '
            , regex=True
        )
        .str.replace(   # Zone 1
            r'[zZ]one (\d+)'
            , r' Zone_\1 '
            , regex=True
        )
        .str.replace(
            r'[cC][/][Ww]'
            , ' consistent_with '
            , regex=True
        )
        .str.replace(
            r'[fF]\/[iI]'
            , ' further-investigation '
            , regex=True
        )
        .str.replace(   # Non specific 관련
            r'\b[Nn][./-][sS]|[nN]on?( other)? [sS]ignificant|without significant change|[nN]o evidence of significant|[nN]on\s*specific|비특이적'
            , ' non_specific '
            , regex=True
        )
        .str.replace(
            r'with or without'
            , r'with_or_without'
            , regex=True
        )
        .str.replace(   # low or high b value
            r'\b(low|high) b value'
            , r' \1_b_value '
            , regex=True
        )
        .str.replace(   # 3 b values
            r'(\d+) b value(s)?'
            , r' \1_cnt b_value '
            , regex=True
        )
        .str.replace(   # (DDx., Rec) 구분 목적의 용어 삭제
            r'\s*(\-|\()?\s*[Dd][Dd][xX].?|'
            r'[rR]ec\s*[\).]'
            , ' '
            , regex=True
        )
        .str.replace(   # d/t, due to
            r'[dD][/][tT]|due to'
            , ' due_to '
            , regex=True
        )
        .str.replace(   # 영상 이미지의 인덱스와 관련된 설명은 전부 삭제.
            r'\([iI][dD][xX]\s*\d+.*?\)\.?'
            , ' '
            , regex=True
        )
        .str.replace(   # 'image' 통일.
            r'imaging'
            , 'image'
            , regex=True
        )
        .str.replace(
            r'A2[/\-]3'
            , r' A2_segment A3_segment '
            , regex=True
        )
        .str.replace(   # A1, A2 등을 segment로 구분
            r'\(?A(\d+)(\s|\.|에|가|\-|\,|\;|$|\)|s)(?:[sS]eg(e)?ment)?s?'
            , r' A\1_segment '
            , regex=True
        )
        .str.replace(   # P1, P2 등을 segment로 구분
            r'P(\d+)(\s|\.|에|\-|\,|\)|$|s)(?:[sS]egment)?s?'
            , r' P\1_segment '
            , regex=True
        )
        .str.replace(
            r'P2\/3'
            , r' P2_segment P3_segment '
            , regex=True
        )
        .str.replace(
            r'V3\/4'
            , r' V3_segment V4_segment '
            , regex=True
        )
        .str.replace(   # M1, M2 등을 segment로 구분
            r'M(\d+)(\s|\.|에|\-||\,|까|$|s)(?:[sS]eg(e)?ment)?s?'
            , r' M\1_segment '
            , regex=True
        )
        .str.replace(   # V1, V2 등을 segment로 구분
            r'V(\d+)(\s|\.|에|\-|\,|의|s|$|\~|\)|\/)(?:[sS]egment)?s?'
            , r' V\1_segment '
            , regex=True
        )
        .str.replace(   # type IV, Bipolar I 등
            r'(type|Bipolar)\s*([IV]+)'
            , r' \1_\2 '
            , regex=True
        )
        .str.replace(
            r'\sC1(\s|$)'
            , r' atlas '
            , regex=True
        )
        .str.replace(
            r'\sC2(\s|$)'
            , r' axis '
            , regex=True
        )
        .str.replace(   # C1 = Atlas, C2 = Axis
            r'C1\,2'
            , r' atlas-axis '
            , regex=True
        )
        .str.replace(   # 뒤에 복수형으로 붙는 약어들
            r'(ICA|ICH|PCA|SDH|SAH|EVD|VA|ACA|MCA|BG|CCA)s'
            , r'\1'
            , regex=True
        )
        .str.replace(   # Axial T1WI, sagittal T1WI, axial T2WI 등
            r'([aA]xial|[Ss]agittal)\s*(T1WI|T2WI|FLAIR|t2-star|DWI)'
            , r' \1-\2 '
            , regex=True
        )
        .str.replace(   # op.bed, op bed 등
            r'op[. ]+bed[., (]|(at )?op bed\.|op[., ]*bed(에서)'
            , ' operative_bed '
            , regex=True
        )
        .str.replace(   # post op
            r'[Pp]ost[ -]*op'
            , r'postop'
            , regex=True
        )
        .str.replace(   # 중이-꼭지돌기염, 유양돌기염
            r'중이-?(꼭지|유양)돌기염'
            , r' otomastoiditis '
            , regex=True
        )
        .str.replace(   # 해면-추체
            r'해면\-추체'
            , r' cavernous_petrous '
            , regex=True
        )
        .str.replace(   # 근위내경동맥 (Proximal Internal Carotid Artery)
            r'근위\s*내경동맥|근위내경돔갱'
            , r' proximal ica '
            , regex=True
        )
        .str.replace(   # 백질-회색질
            r'백질-회색질'
            , r'white_and_gray_matter'
            , regex=True
        )
        .str.replace(
            r'백질|[wW]hite [mM]atter'
            , 'white_matter'
            , regex=True
        )
        .str.replace(
            r'회색질|[gG]ray [mM]atter'
            , 'gray_matter'
            , regex=True
        )
        .str.replace(   # 큰 차이, 큰차이 용어 통일.
            r'큰 차이|큰차이|큰 변화'
            , r' signific_diff '
            , regex=True
        )
        .str.replace(   # toxic/metabolic 등 중간에 '/' 구분 문자 있는 용어 통일.
            r'([a-zA-Z]+)\/([a-zA-Z]+)'
            , r'\1_\2'
            , regex=True
        )
        .str.replace(   # 오타에 의해 [unk] 토큰 분류되는 단어 처리 (위약감).
            r'위얌감[가-힣]*'
            , r' general-weakness '
            , regex=True
        )
        .str.replace(
            r'씰룩\s*거림'
            , r' lip_twitching'
            , regex=True
        )
        .str.replace(   # 영어를 한글로 표기한 것 중 [unk] 토큰 분류되는 단어 처리.
            r'시퀀스'
            , r' sequence '
            , regex=True
        )
        .str.replace(   # benign 오타 수정
            r"beni'gn"
            , r'benign'
            , regex=True
        )
        .str.replace(   # un_rupture_d 오타 수정 unruptred
            r'unruptred'
            , 'unruptured'
            , regex=True
        )
        .str.replace(   # definite 오타 수정
            r'deifnite'
            , r'definite'
            , regex=True
        )
        .str.replace(   # 특정 단어 뒤에 붙은 의미없는 '-' 기호 제거.
            r'(lobe|post |mm)\-(?!\>)'
            , r' \1 '
            , regex=True
        )
        .str.replace(   # 특정 단어 앞에 붙은 의미없는 '-' 기호 제거(2).
            r'\-(insular|positive|T[21]\s|about|diffusion|well|sized|focal)'
            , r' \1'
            , regex=True
        )
        .str.replace(   # MCA, MRA 등의 부위(right, left) 표현 통일.
            r'([RL]|os) (MCA|MRA)'
            , lambda m:
                f'right {m.group(2)}' if m.group(1) == 'R' else
                f'left {m.group(2)}' if m.group(1) == 'L' or m.group(1) == 'os' else ""
            , regex=True
        )
        .str.replace(   # "1.6-m" 등의 cm 오타 정정.
            r'([0-9. \-]+)m\s'
            , r' \1cm'
            , regex=True
        )
    )


## 이미 정형화된 수치 데이터에 대한 마스킹 작업 (중복 변환 방지 목적).
#  Length, Width, Height 등의 정형화된 수치 데이터가 다른 정형화 작업에서 중복으로 변환되지 않도록 한다.
mask_matches = []
def Mask_Repl(match : re.Match) -> str:
    """
    Replacement function used with 're.sub()' to mask text matched by 're.compile()'.
    :param match: matched text object of the regular expression.
    :return: token text(string) to replace the matched text.
    """
    token = f"__PROTECT{len(mask_matches)}__"
    mask_matches.append((token, match.group(0)))    # 정규표현식으로 매칭된 텍스트 원본과 변환(__PROTECT10__ 등)된 값을 저장.
    return token                                    # 변환된 값이 문자열 원문에 반영될 수 있도록 변환 값 리턴.



## 순서(1st, 2nd, 3rd - Ordinal) 및 수량(one, two, three - Cardinal) 표현 정형화 함수.
def cardi_ordinal_preprocessing(df : pd.DataFrame):
    """
    Nomalizing cardinal(e.g. one, two, three) and ordinal(e.g. 1st, 2nd, 3rd) numbers in medical terminology.
    :param text: Medical impression string data (Findings, Conclusion).
    :return: None
    """

    for idx, text in df['context'].items():
        # 1-2th, 7&8th 등 2개 이상의 순서 표현이 혼합된 경우, 앞의 순서 데이터를 먼저 전처리.
        matches = re.findall(r'(\d+)[\-&]+(?=\d+\s*th)', text)
        if matches:
            for v in matches:
                if v == '1':
                    text = re.sub(fr'({v})[\-&]+', r' one_st ', text)
                elif v == '2':
                    text = re.sub(fr'({v})[\-&]+', r' two_nd ', text)
                elif v == '3':
                    text = re.sub(fr'({v})[\-&]+', r' three_rd ', text)
                elif v == '4':
                    text = re.sub(fr'({v})[\-&]+', r' four_th ', text)
                elif v == '5':
                    text = re.sub(fr'({v})[\-&]+', r' five_th ', text)
                elif v == '6':
                    text = re.sub(fr'({v})[\-&]+', r' six_th ', text)
                elif v == '7':
                    text = re.sub(fr'({v})[\-&]+', r' seven_th ', text)
                elif v == '8':
                    text = re.sub(fr'({v})[\-&]+', r' eight_th ', text)
                elif v == '9':
                    text = re.sub(fr'({v})[\-&]+', r' nine_th ', text)

        # 숫자를 포함하여 부위의 번호(ex. 7th)를 표현하는 데이터 전처리.
        matches = re.findall(r'(\d*)(\-*)(\d+(?:th|st|nd|rd))', text)
        if matches:
            for grplist in matches:
                if grplist[0] == '' and grplist[1] == '' and grplist[2]:
                    if grplist[2][0] == '1':
                        text = re.sub(r'1st', ' one_st ', text)
                    elif grplist[2][0] == '2':
                        text = re.sub(r'2nd', ' two_nd ', text)
                    elif grplist[2][0] == '3':
                        text = re.sub(r'3rd', ' three_rd ', text)
                    elif grplist[2][0] == '4':
                        text = re.sub(r'4th', ' four_th ', text)
                    elif grplist[2][0] == '5':
                        text = re.sub(r'5th', ' five_th ', text)
                    elif grplist[2][0] == '6':
                        text = re.sub(r'6th', ' six_th ', text)
                    elif grplist[2][0] == '7':
                        text = re.sub(r'7(th|(?=[ &8-9]+th))', ' seven_th ', text)
                    elif grplist[2][0] == '8':
                        text = re.sub(r'8th', ' eight_th ', text)
                    elif grplist[2][0] == '9':
                        text = re.sub(r'9th', ' nine_th ', text)
                    elif 'th' in grplist[2]:
                        text = re.sub(r'\d+th', ' over_th ', text)

        # lesion의 개수를 전처리.
        matches = re.findall(r'\b(\d+) lesion(s)?', text)
        if matches:
            matches = list(set(matches))
            for grplist in matches:
                if grplist[0] == '1':
                    text = re.sub(fr'{grplist[0]} lesion(s)?', r' one lesion ', text)
                elif grplist[0] == '2':
                    text = re.sub(fr'{grplist[0]} lesion(s)?', r' two lesion ', text)
                elif grplist[0] == '3':
                    text = re.sub(fr'{grplist[0]} lesion(s)?', r' three lesion ', text)
                elif grplist[0] == '4':
                    text = re.sub(fr'{grplist[0]} lesion(s)?', r' four lesion ', text)
                elif grplist[0] == '5':
                    text = re.sub(fr'{grplist[0]} lesion(s)?', r' five lesion ', text)
                elif grplist[0] == '6':
                    text = re.sub(fr'{grplist[0]} lesion(s)?', r' six lesion ', text)
                elif grplist[0] == '7':
                    text = re.sub(fr'{grplist[0]} lesion(s)?', r' seven lesion ', text)
                elif grplist[0] == '8':
                    text = re.sub(fr'{grplist[0]} lesion(s)?', r' eight lesion ', text)
                elif grplist[0] == '9':
                    text = re.sub(fr'{grplist[0]} lesion(s)?', r' nine lesion ', text)

        # 전처리된 text 저장(덮어쓰기)
        df.at[idx, 'context'] = text

    ## '개수'를 의미하는 수치 데이터 전처리.
    df.loc[:, 'context'] = (
        df['context']
        .str.replace(   # (innumerable, > 30)
            r'(innumerable.*?)(\d+)\)'
            , r'\1 \2_cnt '
            , regex=True
        )
        .str.replace(   # 2개
            r'(\d+)\s*개'
            , r' \1_cnt '
            , regex=True
        )
        .str.replace(   # 숫자 + 단어
            r'(\d+)\s*\-?(vessel|patient|in number|faint|small|well\-defined|aneurysms?)'
            , r' \1_cnt \2 '
            , regex=True
        )
    )


## 3, 2, 1차원 길이 표현 ( 2.5 x 1.5 x 0.5 cm | 20x15mm | 12mm 등 )에 대한 Length-Width-Height 정형화 함수
#  cm 단위는 mm 형식으로 맞춘다 (소수점 '.' 표현을 없애기 위한 목적)
#  3차원은 'Length + Width + Height',  2차원은 'Length + Width', 1차원은 'Length'로 표현.
def demention_preprocessing(df : pd.DataFrame) :
    """
    normalizing 3D, 2D and 1D numerical expression in medical terminology.
    :param text: DataFrame
    :return: None
    """

    for idx, text in df['context'].items():
        # 3차원. 변경된 크기 이전의 값을 의미하는 부분의 정형화 (단위 표시가 없으며 뒤에 '-' 또는 '->', '-->', '--->' 등이 붙는다).
        # 실수로 표현된 값은 'cm' 단위이므로 'mm' 단위로 변환한다.
        matches = re.findall(
            r'(\d{1,2}(?:\.\d{1,2})?)\s*(x|\*|X)\s*(\d{1,2}(?:\.\d{1,2})?)\s*(x|\*|X)\s*(\d{1,2}(?:\.\d{1,2})?)\s*(?:mm\s)?(\-{1,4}\>?)'
            , text)
        if matches:
            for grplist in matches:
                # Length에 해당하는 수치 데이터가 실수형인 경우
                if '.' in grplist[0]:
                    Ltmp = str(int(float(grplist[0]) * 10))
                    Lvalue = re.sub(r'\.', r'\\.', grplist[0])
                else:
                    Ltmp, Lvalue = grplist[0], grplist[0]
                # Width에 해당하는 수치 데이터가 실수형인 경우
                if '.' in grplist[2]:
                    Wtmp = str(int(float(grplist[2]) * 10))
                    Wvalue = re.sub(r'\.', r'\\.', grplist[2])
                else:
                    Wtmp, Wvalue = grplist[2], grplist[2]
                # Height에 해당하는 수치 데이터가 실수형인 경우
                if '.' in grplist[4]:
                    Htmp = str(int(float(grplist[4]) * 10))
                    Hvalue = re.sub(r'\.', r'\\.', grplist[4])
                else:
                    Htmp, Hvalue = grplist[4], grplist[4]

                # Length 정형화
                text = re.sub(
                    fr'([^1-9]|^|\(){Lvalue}(?=\s*(x|\*|X)\s*{Wvalue}\s*(x|\*|X)\s*{Hvalue}' + r'\s*\-{1,4}\>?)'
                    , fr' Length-{Ltmp}mm '
                    , text)

                # Width 정형화
                text = re.sub(fr'(?<=Length\-{Ltmp}mm).+{Wvalue}(?=\s*(x|\*|X)\s*{Hvalue}' + r'\s*\-{1,4}\>?)'
                              , fr' Width-{Wtmp}mm '
                              , text)

                # Height 정형화
                text = re.sub(fr'(?<=Length\-{Ltmp}mm Width\-{Wtmp}mm).+{Hvalue}\s*(mm)?' + r'\s*\-{1,4}\>?'
                              , fr' Height-{Htmp}mm change '
                              , text)
                # print(matches)
                # print(Ctext)

        # 3차원 크기 데이터 정형화
        matches = re.findall(
            r'(\d{1,2}(?:\.\d{1,2})?)\s*(?:cm|mm)?(x|\*|X)(?:\.)?\s*(\d{1,2}(?:\.\d{1,2})?)\s*(?:cm|mm)?(x|\*|X)(?:\.)?\s*(\d{1,2}(?:\.\d{1,2})?)\s*(?:\(neck\))?\s*(cm|mm| )'
            , text)
        if matches:
            for grplist in matches:  # 매칭된 그룹 리스트 순환. 6개의 원소가 하나의 그릅에 포함.
                if grplist[-1] == 'cm':  # cm 단위라면 mm 단위로 변환 (cm 단위는 소수점이 포함되지만, mm 단위는 정수만으로 표현 가능).
                    Ltmp = str(int(float(grplist[0]) * 10))  # 정형화 값으로 사용할 mm단위의 L,W,H 값.
                    Wtmp = str(int(float(grplist[2]) * 10))
                    Htmp = str(int(float(grplist[4]) * 10))
                    Lvalue = re.sub(r'\.', r'\\.', grplist[0])  # Length Value : 특수문자 '.'를 정규표현식에서 일반 문자로 보이도록 변경.
                    Wvalue = re.sub(r'\.', r'\\.', grplist[2])  # Width Value.
                    Hvalue = re.sub(r'\.', r'\\.', grplist[4])  # Height Value.

                    # Length 정형화
                    text = re.sub(
                        fr'{Lvalue}' + r'(?=\s*(cm)?(x|\*|X)\.?\s*(\d{1,2}(\.\d{1,2})?)\s*(cm)?(x|\*|X)\.?\s*(\d{1,2}(\.\d{1,2})?)\s*cm)'
                        , fr' Length-{Ltmp}mm '
                        , text)
                    # Width 정형화
                    text = re.sub(fr'(?<=Length-{Ltmp}mm ).+{Wvalue}\s*(cm)?(?=(x|\*|X)\.?\s*{Hvalue})'
                                  , fr'Width-{Wtmp}mm '
                                  , text)
                    # Height 정형화
                    text = re.sub(fr'(?<=Length-{Ltmp}mm Width-{Wtmp}mm ).+{Hvalue}\s*cm'
                                  , fr'Height-{Htmp}mm '
                                  , text)
                else:
                    text = re.sub(fr'{grplist[0]}[ *xX]*(?={grplist[2]}[ *xX]*{grplist[4]}\s*(\(neck\))?[ m]*)'
                                      , fr' Length-{grplist[0]}mm '
                                      , text)
                    text = re.sub(
                            fr'(?<=Length-{grplist[0]}mm ){grplist[2]}[ xXm*]*(?={grplist[4]}\s*(\(neck\))?[ m]*)'
                            , fr'Width-{grplist[2]}mm '
                            , text)
                    text = re.sub(
                            fr'(?<=Length-{grplist[0]}mm Width-{grplist[2]}mm )[ *xXm]*{grplist[4]}\s*(\(neck\))?[ m]*'
                            , fr'Height-{grplist[4]}mm '
                            , text)
                # print(matches)
                # print(Ftext)

        # 2차원. 변경된 크기 이전의 값을 의미하는 부분의 정형화 (단위 표시가 없으며 뒤에 '-' 또는 '->', '-->', '--->' 등이 붙는다).
        # 실수로 표현된 값은 'cm' 단위이므로 'mm' 단위로 변환한다.
        matches = re.findall(
                r'(\d{1,2}(?:\.\d{1,2})?)\s*(x|\*|X)\s*(\d{1,2}(?:\.\d{1,2})?)\s*(?:cm|mm)?\s*(\-{1,4}\>*)(\s*<)?(?!cm|mm)'
                , text)
        if matches:
            for grplist in matches:
                # Length에 해당하는 수치 데이터가 실수형인 경우
                if '.' in grplist[0]:
                    Ltmp = str(int(float(grplist[0]) * 10))
                    Lvalue = re.sub(r'\.', r'\\.', grplist[0])
                else:
                    Ltmp, Lvalue = grplist[0], grplist[0]
                # Width에 해당하는 수치 데이터가 실수형인 경우
                if '.' in grplist[2]:
                    Wtmp = str(int(float(grplist[2]) * 10))
                    Wvalue = re.sub(r'\.', r'\\.', grplist[2])
                else:
                    Wtmp, Wvalue = grplist[2], grplist[2]

                # Length 정형화
                text = re.sub(
                    fr'([^1-9]|^|\(){Lvalue}' + r'(?=\s*(x|\*|X)\s*\d{1,2}(\.\d{1,2})?\s*(cm|mm)?\s*\-{1,4}\>*)(\s*<)?(?!cm|mm)'
                    , fr' Length-{Ltmp}mm '
                    , text)

                # Width 정형화
                text = re.sub(fr'(?<=Length\-{Ltmp}mm ).+{Wvalue}\s*(cm|mm)?' + r'\s*\-{1,4}\>*(\s*<)?'
                              , fr'Width-{Wtmp}mm change '
                              , text)
            # print(matches)
            # print(Ftext)

        # 2차원 크기 데이터 정형화
        matches = re.findall(
            r'(\d{1,2}(?:\.\d{1,2})?)\s*(?:cm|mm)?(x|\*|X|\&)(?:\.)?\s*(\d{1,2}(?:\.\d{1,2})?)\s*(?:\-)?(cm|mm|\))'
            , text)
        if matches:
            matches = list(set(matches))
            for grplist in matches:
                if grplist[-1] == 'cm' or grplist[-1] == ')':
                    Ltmp = str(int(float(grplist[0]) * 10))
                    Wtmp = str(int(float(grplist[2]) * 10))
                    Lvalue = re.sub(r'\.', r'\\.', grplist[0])
                    Wvalue = re.sub(r'\.', r'\\.', grplist[2])

                    # Length 정형화
                    text = re.sub(
                            fr'([^1-9]|^){Lvalue}' + r'(?=\s*(cm)?(x|\*|X)\.?\s*(\d{1,2}(\.\d{1,2})?)\s*\-?(cm|\)))'
                            , fr' Length-{Ltmp}mm '
                            , text)
                    # Width 정형화
                    text = re.sub(fr'(?<=Length\-{Ltmp}mm ).+{Wvalue}\s*\-?(cm|\))'
                                      , fr'Width-{Wtmp}mm '
                                      , text)
                else:
                    # mm 단위인데 실수 형태인 경우, 반올림하여 정수화.
                    if grplist[-1] == 'mm' and '.' in grplist[0]:
                        Ltmp = str(round(float(grplist[0])))
                        Lvalue = re.sub(r'\.', r'\\.', grplist[0])
                        text = re.sub(
                            fr'([^1-9]|^){Lvalue}' + r'(?=\s*(mm)?(x|\*|X)\.?\s*(\d{1,2}(\.\d{1,2})?)\s*\-?mm)'
                            , fr' Length-{Ltmp}mm '
                            , text)
                    else:
                        # Length 정형화
                        text = re.sub(
                            fr'([^1-9]|^){grplist[0]}' + r'(?=\s*(mm)?(x|\*|X|\&)\.?\s*(\d{1,2}(\.\d{1,2})?)\s*\-?mm)'
                            , fr' Length-{grplist[0]}mm '
                            , text)

                    if grplist[-1] == 'mm' and '.' in grplist[2]:
                        Wtmp = str(round(float(grplist[2])))
                        Wvalue = re.sub(r'\.', r'\\.', grplist[2])
                        text = re.sub(fr'(?<=Length\-{Ltmp}mm ).+{Wvalue}\s*\-?mm'
                                          , fr'Width-{Wtmp}mm '
                                          , text)
                    else:
                        # Width 정형화
                        text = re.sub(fr'(?<=Length\-{grplist[0]}mm ).+{grplist[2]}\s*\-?mm'
                                        , fr'Width-{grplist[2]}mm '
                                        , text)
                # print(matches)
                # print(Ctext)

        # 1차원 크기 데이터 정형화 (3차원, 2차원 크기 데이터에 대해 모두 정형화가 완료된 상태여야 한다.)
        # 이미 정형화가 완료된 텍스트는 중복 변환되지 않도록 마스킹 처리 후 마지막에 복원하는 방법으로 구현.
        global mask_matches
        mask_matches = []
        mask_pattern = re.compile(r'\b(?:Length|Width|Height)\-\d{1,3}(?:\.\d{1,3})?mm\b')
        masked = mask_pattern.sub(Mask_Repl, text)  # 정형화 전 이미 정형화된 데이터는 겹치지 않도록 마스킹.

        # 마스킹된 텍스트에서 크기 변경 전 1차원 크기 데이터 추출 (ex. 1.5 - 2.1cm 형태에서 1.5 값)
        matches = re.findall(r'(\d+(?:\s*\.\d+)?)\s*(?:cm|mm)?\s*(\-{1,5}>?)[ 0-9.]*?(cm|mm)', masked)
        if matches:
            # 중복되는 Group이 2번 이상 정형화되지 않도록 세트화.
            matches = sorted(list(set(matches)), key=lambda x: float(x[0]), reverse=True)
            for grplist in matches:
                # 2.2-cm, 20-mm 등의 단일 값 처리
                if grplist[1] == '-':
                    if grplist[-1] == 'cm':
                        Ltmp = str(int(float(grplist[0]) * 10))
                        Lvalue = re.sub(r'\.', r'\\.', grplist[0])
                        masked = re.sub(fr'{Lvalue}\-cm', fr' Length-{Ltmp}mm ', masked)
                    elif grplist[-1] == 'mm':
                        if '.' in grplist[0]:
                            Ltmp = str(int(round(float(grplist[0]))))
                            Lvalue = re.sub(r'\.', r'\\.', grplist[0])
                            masked = re.sub(fr'{Lvalue}\-mm', fr' Length-{Ltmp}mm ', masked)
                        else:
                            masked = re.sub(fr'{grplist[0]}\-mm[.,]?', fr' Length-{grplist[0]}mm ', masked)
                    # 크기 변동 이전의 값 처리
                elif '>' in grplist[1]:
                    if grplist[-1] == 'cm':
                        Ltmp = str(int(float(grplist[0]) * 10))
                        Lvalue = re.sub(r'\.', r'\\.', grplist[0])
                        masked = re.sub(fr'{Lvalue}\s*(cm)?\s*\-+\>', fr' Length-{Ltmp}mm change ', masked)
                    elif grplist[-1] == 'mm':
                        if '.' in grplist[0]:
                            Ltmp = str(int(round(float(grplist[0]))))
                            Lvalue = re.sub(r'\.', r'\\.', grplist[0])
                            masked = re.sub(fr'{Lvalue}\s*\-+\>', fr' Length-{Ltmp}mm change ', masked)
                        else:
                            masked = re.sub(fr'{grplist[0]}\s*\-+\>', fr' Length-{grplist[0]}mm change ', masked)

                # 마스킹된 텍스트를 복원 후, 최종 결과를 text에 저장.
                for token, tmp in mask_matches:
                    masked = masked.replace(token, tmp)

                text = masked
                masked = mask_pattern.sub(Mask_Repl, text)

        mask_matches = []
        mask_pattern = re.compile(r'\b(?:Length|Width|Height)\-\d{1,3}(?:\.\d{1,3})?mm\b')
        masked = mask_pattern.sub(Mask_Repl, text)

        # 마스킹된 텍스트에서 1차원 크기 데이터 추출.
        matches = re.findall(r'(\d+(?:[.,]\d+)?)\s*(mm|cm)[가-힣]*\b', masked)
        if matches:
            # 중복되는 Group에 의해 2번 정형화되지 않도록 세트화.
            converted = [(s.replace(',', '.'), unit) for s, unit in matches]  # ','로 오타가 난 실수 데이터를 '.'로 변환.
            matches = sorted(list(set(converted)), key=lambda x: float(x[0]),
                             reverse=True)  # 실수 값을 기준으로 내림차순 정렬 (의도되지 않은 중복 정제 오류 방지 목적).
            for grplist in matches:
                if grplist[-1] == 'cm':
                    Ltmp = str(int(float(grplist[0]) * 10))
                    Lvalue = re.sub(r'\.', r'\\.', grplist[0])
                    masked = re.sub(fr'([^1-9]|^|\(){Lvalue}\s*cm', fr' Length-{Ltmp}mm ', masked)
                elif grplist[-1] == 'mm' and '.' in grplist[0]:
                    # Ltmp = re.sub(r'(\d{1,2}).+', r'\1', grplist[0])
                    Lvalue = re.sub(r'\.', r'\\.', grplist[0])
                    Ltmp = str(round(float(grplist[0])))
                    masked = re.sub(fr'((?<!\d)|^|\(|(?<!Length\-))({Lvalue}\s*mm)', fr' Length-{Ltmp}mm ', masked)
                elif grplist[-1] == 'mm' and ',' in grplist[0]:
                    Ltmp = str(round(float(grplist[0].replace(',', '.'))))
                    masked = re.sub(fr'((?<!\d)|^|\(|(?<!Length\-))({grplist[0]}\s*mm)', fr' Length-{Ltmp}mm ', masked)
                else:
                    masked = re.sub(fr'((?<!\d)|^|\(|(?<!Length\-)){grplist[0]}\s*mm[가-힣]*',
                                    fr' Length-{grplist[0]}mm ',
                                    masked)

                # 마스킹된 텍스트를 복원 후, 최종 결과를 Ctext에 저장.
                for token, tmp in mask_matches:
                    masked = masked.replace(token, tmp)

                text = masked
                masked = mask_pattern.sub(Mask_Repl, text)

        #  두 크기(길이) 값 사이에 존재하는 '변동'을 의미하는 특수 문자의 정형화 처리.
        text = re.sub(r'(?<=mm)\s*\-+\>\s*(?=Length\-)', ' change ', text)

        # 전처리된 text 저장(덮어쓰기)
        df.at[idx, 'context'] = text



## 날짜, 시간, 영상 인덱스 번호 등, 구분 주제(Note, DDX, e.g. 등) 문자열 전처리 함수.
def unnecessary_preprocessing(df : pd.DataFrame):
    """
    Function to remove unnecessary string data
    :param text: DataFrame
    :return: None
    """

    df.loc[:, 'context'] = (
        df['context']
        # 날짜 기록 데이터 (2011.07.08.), (2011. 11. 11.), (2004) 전체 삭제.
        # 날짜 데이터는 '이전'의 의미를 전달할 뿐, 크게 의미 있지 않다고 판단하여 텍스트 삭제 진행.
        .str.replace(
            r'\(?\d{2,4}[. ]+\d{1,2}[. ]+\d{1,2}[. ]+?\)?\\?|'      # (14.02.15.)
            r'\(\d{4}\)|\(\d{1,2}\/\d{1,2}\)\.|'                    # (5/19).                          
            r'\(?\d{4}([.\- ]*\d{1,2}[.\- ]*\d{1,2}[.\-), ]*)+|'    # (2020-09-09, 09-21)
            r'on (\d+\/\d+\/\d{4}|\d{1,2}\/\d{1,2})\)?\.?|'         # on 5/8/2024, on 5/16
            r'on \d{4}[\-.]\d{1,2}[\-.]\d{1,2}[).,]*|'              # on 2021.5.28,  on 2022-01-05).
            r'in \d{4}(\)\.)|'                                      # in 2024).
            r'\d+년\s*\d{1,2}월\s*\d{1,2}일'                           # 2020년 6월 10일
            r'밤 \d+시경|'                                             # 밤 11시경
            r'[0-9 \-]+[0-9]일|'                                     # 6 - 9일
            r'for[ 0-9]+days?|'                                     # for 2 days
            r'in \d{4}[ .]+\d{1,2}[. ]+\d{1,2}[., ]+'               # in 2017.9.13,
            , ' '
            , regex=True
        )
        .str.replace(   # 특정 구분 텍스트 삭제 (DDX, Note 등), 영상 이미지 인덱스 번호 표현  ([IDX 4 IM 17].), (Se1, Im 15)
            r'\-\s*[iI][dD][xX][0-9 ,]*[iI][mM][0-9 ,]*[iI][dD][xX][0-9 ,]*[iI][mM][0-9 ,]*[.,\- ]?|'
            r'[\[ ]*I[Dd][Xx][ 0-9]*I[Mm][ 0-9,-]*[\] ,]*|'
            r'\([Ii][Dd][Xx][. 0-9]+image \d+\s*\)|'
            r'\[stack.*?IDX[ 0-9-]*IM[ 0-9-*\]]*\.?|'
            r'\([sS]e\d+[ ,]*I[Mm][ 0-9]*\)|'
            r'\s[iI][Mm][ 0-9\).,\]]+|'
            r'\(\#[iI][Dd][Xx][0-9 ,.\)\-\/]+|' 
            r'\(\d( |\,|and)*\d( |\,|and)*\d\)'
            , ' '
            , regex=True
        )
        .str.replace(   # 'N년 전'의 의미를 갖는 문자열 전처리.
            r'(\d+)\s*(yrs?|years?|months?)\s*ago'
            , lambda m:     # 2 yr ago  = 2-year-before
                f' year_{m.group(1)}_be ' if m.group(2) in ['yr', 'yrs', 'year', 'years'] else
                f' month_{m.group(1)}_be ' if m.group(2) in ['month', 'months'] else ""
            , regex=True
        )
        .str.replace(   # 'after N-N years' 등의 범위로 작성된 문자열 전처리.
            r'after\s*\d+[-~](\d+)\s*years?'
            , r' year_\1_af '                   # after 1~2 years
            , regex=True
        )
        .str.replace(   # 'after N-N months' 등의 범위로 작성된 문자열 전처리.
            r'after\s*\d+[-~](\d+)\s*months?'
            , r' month_\1_af '                  # after 6-12 months
            , regex=True
        )
        .str.replace(   # 'after N-N weeks' 등의 범위로 작성된 문자열 전처리.
            r'after\s*\d+[-~](\d+)\s*weeks?'
            , r' week_\1_af '                   # after 4-8 weeks
            , regex=True
        )
        .str.replace(   # 'N년 이후에'의 의미를 갖는 문자열 전처리.
            r'(?:after|\>)\s*(\d+)\s*(yrs?|years?|months?)\.?'
            , lambda m:     # after 1 year  = 1-year-after
                f' year_{m.group(1)}_af ' if m.group(2) in ['yr', 'yrs', 'year', 'years'] else
                f' month_{m.group(1)}_af ' if m.group(2) in ['month', 'months'] else ""
            , regex=True
        )
        .str.replace(
            r'(\*{2}[ 1-9\-,.]+)|'                  # ** 1-2, **1,2
            r'(\bI+\.\s)|'                          # I., II., III.,
            r'(\*?\s*[nN]ote\s*[,:.])|'             # * Note:
            r'((?<=[a-zA-Z가-힣\)])\.(\s|$|\n|\t))|' # 문장의 마지막 '.'
            r'((\b|^)(\d|10)\s*[.,](\s|(?=MRA|Both|Right|Diff|Mic))|\d\s*(?=No -significant))|'  # 1., 2., 3.,
            r'([(\[]\d[)\]]\:?)|'                   # (1), (2), [1], [2] ...
            r'((?<=\w)[`\'\’]([Ss]|\s)*)|'          # Parkinson's
            r'(\s(\-+|\(|\))\s)|'                   # 구분 문자 역할의 ' - ' 등.
            r'(((\s|^|\*+)\d\))+[.,]?\s)|'          # 순서 번호 역할의 1), 2).
            r'(e\.g\.?)|'                           # e.g = for example
            r'(Ex\))|'                              # Ex)
            r'(박정식\.)'                            # 무의미한 사람 이름 포함.
            , r' '
            , regex=True
        )
        .str.replace(   # left 약어 처리.
            r'(\(|\b)?[lL]t\s'
            , ' left '
            , regex=True
        )
        .str.replace(   # right 약어 처리.
            r'(\s|\()[rR]t[ )]'
            , ' right '
            , regex=True
        )
        # 대소 비교 문자('<', '>') 텍스트 변환.
        # '<', '>'는 다른 특수 문자와 조합하여 '구분 문자' 역할로 사용하는 경우도 있다.
        # 이를 제외(삭제)하면 크기나 백분율 비교로 사용하므로 이들에 대한 명확한 텍스트 변환이 필요하다.
        # 대소 구분 목적으로 사용하는 기호 표현을 'less than', 'greater than'으로 텍스트 변경한다.
        .str.replace(
            r'Lt\.?\s*\>\s*Rt\.?|Rt\.?\s*\<\s*Lt\.?|[lL]eft\s*\>\s*[rR]ight|[rR]ight\s*\<\s*[lL]eft'
            , ' left_greater_than_right '
            , regex=True
        )
        .str.replace(
            r'Rt\.?\s*\>\s*Lt\.?|Lt\.?\s*\<\s*Rt\.?|[rR]ight\s*\>\s*([lL]eft|[lL]t)|[lL]eft\s*\<\s*[rR]ight'
            , ' right_greater_than_left '
            , regex=True
        )
        .str.replace(
            r'(?<!temporal)(?<!MRA)(?<!\-)(?<!ings)>\s*(?=[rR]ight|[Ll]eft|[rR]t|[lL]t|\d|[gG]rade|Length)|greater than'
            , ' greater_than '
            , regex=True
        )
        .str.replace(   # (< Length-5mm, (< Grade, ...
            r'\(?<\s*(?=[rR]ight|[Ll]eft|[rR]t|[lL]t|\d|[gG]rade|Length)|less than'
            , ' less_than '
            , regex=True
        )
        .str.replace(   # <Brain, dings>
            r'<(?=[a-zA-Z가-힣 ])|(?<=[a-zA-Z가-힣 *])>'
            , ' '
            , regex=True
        )
        .str.replace(   # DWI,
            r'(?<=[a-zA-Z가-힣])\s*[,:;]'
            , ' '
            , regex=True
        )
        .str.replace(  # 구분자 역할의 특수문자 제거
            r'\((?=[a-zA-Z])|(?<=[a-zA-Z])\)|'
            r'[-=]+>|'
            r'\(<-+|(?<=n) <- (?=s)|'
            r'\*\.\s*\*\s*|'
            r'\*+\d+[, ]\d*|'
            r'\d'
            , ' '
            , regex=True
        )
        .str.replace(   # '분', '환자', 'pt'를 'patient'로 통일.
            r'환자\s*(분)?|\b(분|pt)\b'
            , r' patient '
            , regex=True
        )
    )


## 양성(positive), 음성(negative) 내용 전처리.
def pos_neg_preprocessing(df : pd.DataFrame):
    """
    Preprocessing of positive and negative contents.
    :param text: DataFrame
    :return: None
    """
    df.loc[:, 'context'] = (
        df['context']
        .str.replace(
            r'\(\+\)|\w\s\+'
            , 'positive'
            , regex=True
        )
        .str.replace(
            r'\(\-\)'
            , 'negative'
            , regex=True
        )
        .str.replace(
            r'(MRA)\s*\:\s*\+'
            , r' \1 positive '
            , regex=True
        )
        .str.replace(
            r'(MRA)\s*\:\s*\-$'
            , r' \1 negative '
            , regex=True
        )
    )


## '__SEP__' 토큰 재정비.
def special_token_preprocessing(df : pd.DataFrame):
    # 문장 첫 시작은 [CLS] token
    df['context'] = '[CLS] ' + df['context'] + ' [SEP]'

    df.loc[:, 'context'] = (
        df['context']
        .str.replace(   # '__SEP__' 토큰 값을 '[SEP]'로 변환
            r'__SEP__'
            , r'[SEP]'
            , regex=True
        )
    )

    # 토큰들이 겹치지 않고 1개만 남을 때까지 반복 작업
    # ex.  "[SEP] [SEP] [SEP]"로 된 경우, 1번만 정규표현식이 동작하면 "[SEP] [SEP]"가 되므로 2번 이상의 작업이 필요하다.
    for idx, text in df['context'].items():
        while True:
            new_text = re.sub(r'(?:\[(SEP|CLS)\])[ ]+\[SEP\]', r'[\1]', text)
            if new_text == text:
                break
            text = new_text
        #전처리한 문장 저장
        df.at[idx, 'context'] = text


# df['tokens'] Column을 생성 후, word tokenize() 결과를 담는다.
# 토큰화 과정에서는 2개의 토크나이저를 사용한다.
# 처음으로 한국어 반영 토크나이저로 주요 단어들을 토큰화 하고, 해당 토큰이 다음 토크나이저의 사전에 없으면 추가한다.
# 다음으로 사용할 토크나이저는 의학 용어가 잘 반영되는 모델을 사용한다.
def word_tokenizing(df : pd.DataFrame):
    # 토큰화 결과를 저장한 신규 컬럼 생성
    df['tokens'] = [[]] * len(df)
    # 한글 호환되는 토크나이저
    kor_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    kor_tokenizer.add_tokens(['폐쇄', '찢', '대뇌', '소뇌', '두통', '수술',
                              '감소', '증가', '관찰'])  # [UNK]로 분류되는 단어 또는 의학 용어 추가.



    # 먼저, 한국어 호환되는 토크나이저의 토큰화 결과를 담는다.
    for idx, text in df['context'].items():
        kts = kor_tokenizer.tokenize(text)

        words = []          # 의학 토큰화 토크나이저(tokenizer_bert)의 vocab에 추가할 단어들 저장.
        current_word = ''   # 추가될 단어
        # 토큰화된 결과 중 '##'으로 이어 붙는 단어들을 1개의 단어로 통합한다.
        for token in kts:
            if token.startswith('##'):
                current_word += token[2:]
            else:
                if current_word:
                    words.append(current_word)
                current_word = token
        if current_word:
            words.append(current_word)

        # 생성한 토큰들 중 '한글'로 구성된 단어들만 추출한다.
        hangul_only = [token for token in words if re.fullmatch(r'[가-힣#]+', token)]
        # 겹치는 단어는 제외한다.
        new_tokens = list(set(hangul_only))
        # tokenizer_bert에 없는 단어들만 추출한다.
        tokens_to_add = [tok for tok in new_tokens if tok not in tokenizer_bert.get_vocab()]
        # vocab에 추가한다.
        tokenizer_bert.add_tokens(tokens_to_add)


    # 2차, tokenizer_bert로 토큰화한 결과 중 '_'를 포함하는 조합의 경우 1개의 단어로 통합할 수 있도록 사전 추가
    for idx, text in df['context'].items():
        tts = tokenizer_bert.tokenize(text)

        current_word = ''   # 현재 단계의 단어
        prev_word = ''      # 이전 단계의 단어
        for token in tts:
            if token.startswith('_') or prev_word.endswith('_'):
                current_word += token
                prev_word = current_word
            else:
                if current_word and \
                    current_word not in tokenizer_bert.vocab:
                        tokenizer_bert.add_tokens([current_word])
                current_word = token
                prev_word = current_word
        if current_word and current_word not in tokenizer_bert.vocab:
            tokenizer_bert.add_tokens([current_word])


    # tokenizer_bert로 토큰화된 리스트를 df['tokens']에 저장.
    df['tokens'] = df['context'].apply(lambda x: tokenizer_bert.tokenize(x))

    #raw_sent = df['context'].tolist()
    # 불용어 제거 과정 없이 임베딩 작업 시
    # encoded = tokenizer_bert.batch_encode_plus(
    #     raw_sent,
    #     padding=True,
    #     truncation=True,
    #     max_length=512,
    #     return_tensors='pt',
    #     return_attention_mask=True
    # )



# 토큰 결과에서 불용어 단어/특수문자 제거.
def stopword_removal(df : pd.DataFrame):
    # special token을 제외한 모든 토큰을 소문자 변환.
    df['tokens'] = df['tokens'].apply(
        lambda token_list : [tk if tk in ['[CLS]', '[SEP]']
                             else tk.lower()
                             for tk in token_list]
    )

    # stopwords에 포함된 단어는 토큰에서 제외
    df['tokens'] = df['tokens'].apply(
        lambda token_list : [tk for tk in token_list if tk not in stopwords]
    )

    max_token_size = df['tokens'].apply(len).max()
    return max_token_size



# input_ids에 대한 패딩과 Attention Mask를 추가.
def padding_attention(input_ids : list, MAX_LEN : int):
    # 모든 input_ids의 크기가 512보다 작음을 확인.
    padd_len = MAX_LEN - len(input_ids)

    # 패딩 추가 ([PAD] 토큰은 0의 값을 가짐)
    padded_ids = input_ids + [0] * padd_len

    # Attention mask 생성
    attention_ids = [1] * len(input_ids) + [0] * padd_len

    return padded_ids, attention_ids


def embedding(df : pd.DataFrame):
    # Embedding 작업
    df['input_ids'] = df['tokens'].apply(tokenizer_bert.convert_tokens_to_ids)

    # padding 추가 및 attention mask 생성
    MAX_LEN = 512
    df[['padd_ids', 'att_ids']] = df['input_ids'].apply(
        lambda input_ids : pd.Series(padding_attention(input_ids, MAX_LEN))
    )




def training(train_dataloader : DataLoader):
    # 학습 돌릴 cpu/gpu 선택
    device = Checking_cuda()

    # BERT 학습 모델.
    # 먼저 구성 객체 설정.
    config = BertConfig.from_pretrained(
        'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
        #'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
    num_labels=2
    )

    # 분류용 헤드를 수동으로 생성.
    model = BertForSequenceClassification.from_pretrained(
        'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
        #'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
        config=config
    )

    gc.collect()
    torch.cuda.empty_cache()

    # 모델을 학습 모드로 두고 진행.
    model.train()
    # 모델을 gpu에 담기.
    model.to(device)
    # 토크나이저 단어 사전에 사용자 추가된 것이 있으므로 개수 반영.
    model.resize_token_embeddings(len(tokenizer_bert))
    # 옵티마이저
    optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)

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



## 학습한 모델을 테스트
def validating(model : BertForSequenceClassification
               , test_dataloader : DataLoader):

    device = Checking_cuda()
    valmodel = model

    valmodel.to(device)
    valmodel.eval()

    pred_list = []
    label_list = []

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

    # 정확도, 정밀도, F1-score 등의 통계
    print(f'테스트 통계\n{classification_report(pred_list, label_list)}')

    # AUC 결과 (C-statistic)
    print(f"정확도(Accuracy): {round(roc_auc_score(pred_list, label_list), 6)}")





## 테스트 목적의 csv 파일 반환.
# Raw_Text : 텍스트 데이터 원본.
# After_Text : Preprocessing 결과 텍스트.
def Get_DataFrame_to_CSV(raw_txt, af_txt):
    """
    Exporting the dataframe to a CSV file.
    :param raw_txt: raw(original) text.
    :param af_txt: converted text.
    :return: None
    """
    df = pd.DataFrame({
        'Raw_Text': raw_txt,
        'After_Text': af_txt
    })

    df.to_csv('output.csv', index=False, encoding='utf-8-sig')  # 윈도우에서 한글 포함 시 utf-8-sig 권장



## GPU 또는 CPU 사용 가능한지 테스트 후 반환
def Checking_cuda():
    print(torch.__version__)        # PyTorch 버전
    print(torch.version.cuda)       # 내장된 CUDA 버전
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('%d GPU(s) available.' % torch.cuda.device_count())
        print('Can use the GPU:', torch.cuda.get_device_name(0))
        print("CUDA Version:", torch.version.cuda)
    else:
        device = torch.device("cpu")
        print('No GPU available, using the CPU instead.')

    return device