"""
- TrainCopySet.csv : 원본 데이터 Set
-
- tensorflow 사용 시 Python3.9 버전에서는 2.11.0 Version을 사용한다  -->  Module Import 오류가 발생한다.
- python 3.9 이상의 버전에서 pytorch 설치 명령 링크 : https://pytorch.org/get-started/locally/
- 학습용 데이터셋은 'TrainSet _1차.csv', 검증(테스트) 데이터셋은 'ValidationSet_2차.csv'를 사용.

"""
import pandas as pd                         # DataFrame
import re                                   # Regular Expression
import torch, gc                            # Pytorch, Garbage collection
import nltk                                 # Word Tokenization
from transformers import BertTokenizer, AutoTokenizer
from transformers import get_linear_schedule_with_warmup    # 학습률 조정 스케줄러
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score
from nltk.tokenize import sent_tokenize     # 문장 자연어 토큰화
from tensorflow.keras.preprocessing.sequence import pad_sequences  #Keras 시퀀스
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertConfig
from torch.optim import AdamW


# Resource File Download
nltk.download('punkt')      # 구두점 분리가 학습된 모델 (분리 규칙 모델)
#nltk.download('punkt_tab')
nltk.download('stopwords')  # 불용어 사전


## 토큰화 사전에 없는 용어(UNK) 추가
#  이후에도 Token Embedding 작업에 사용.
'''
토크나이저 테스트 종류
 - bert-base-multilingual-cased : 다국어 지원 토크나이저
 - microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract : 의학 소견/논문 바탕의 pretrained model
 - microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract
'''
vocab = ['찢어지는', '촤측', '없', "폐쇄", '높', '빠', '쪽', '쓰', '낮', '양', '받', '괜찮', '앞', '눈']
tokenizer_bert = BertTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')
tokenizer_bert.add_tokens(vocab)

## 불용어 토큰 제거 Tuple.
#  stopwords외에 별도로 처리할 목적으로 사용자 정의.
stopwords = ('and', 'at', 'a', 'an', 'as', 'are', 'b', 'in', 'to', 'the', 'of', 'or',
             'm', 's',
             '가', '과', '그', '그리고', '는', '년', '등의', '또는', '로', '및', '볼', '분', '수', '서',
             '에서', '이에', '이', '인', '의', '외', '와', '을', '인해', '에는', '에', '에도',
             '중', '지', '현', '함', '~', '"',
             '.', ',', ':', '(', ')', '→', '[', ']', '/', '*', '=', '+', "'", '&', '#', '?', ';')

# 하나의 문장에 유의미한 최대 토큰 개수 (최대 512개의 임베딩이 가능)
max_token_size = 0

# 토크나이저, 모델 저장 경로
model_save_path = '../../saved_bert_model_5'


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
    #print(f"Findings 결측값 = {df['Findings'].isnull().sum()}")
    #print(f"Conclusion 결측값 = {df['Conclusion'].isnull().sum()}")

    # 모든 결측값에 'no-text' 대체 단어 추가.
    df['Findings'].fillna('empty finding.', inplace=True)
    df['Conclusion'].fillna('empty conclusion', inplace=True)

    df.fillna(' ', inplace=True)
    print("@@@@ 결측값(NaN)을 빈 문자열('') 처리한다. @@@@\n -- 처리 결과 -- ")
    # 결측치 처리 결과
    print(f"Findings 결측값 처리 후 = {df['Findings'].isnull().sum()}")
    print(f"Conclusion 결측값 처리 후 = {df['Conclusion'].isnull().sum()}")
    print('-----------------------------------------------------------')


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



## 대뇌의 4개(Parietal, Temporal, Occipital, Frontal)의 엽(Lobe) 분류 용어에 대한 정형화.
def lobe_preprocessing(text : str) -> str:
    """
    Convert the names of the four cerebral lobes into a single token format.
    @Frontal lobe = 전두엽(운동, 판단, 언어)
    @Parietal lobe = 두정엽(공간 지각, 감각 정보)
    @Temporal lobe = 측두엽(청각, 기억)
    @Occipital lobe = 후두엽(시각 처리)
    :param text: Medical impression string data (Findings, Conclusion).
    :return: Converted string data.
    """
    #  both, bilateral 'parietal' + 'temporal' + 'occipital'
    text = re.sub(r'[Bb]oth\s*[PTO]\-[PTO]\-[PTO]\s*(lobe(s)?)?|'
                  r'(at )?(the )?([Bb]ilateral|[Bb]oth|양측) (temporal|parietal|occipital)[ &,-]*(temporal|parietal|occipital)(\,|\s|\&|and|\-)*(temporal|parietal|occipital)\s*(lobe(s)?|(\.|\,|and|\s)*(?=left|right))|'
                  r'(at )?(the )?([Bb]ilateral|[bB]oth|양측) (parieto|temporo|occipito)[ \-,]*(parieto|temporo|occipito)[ \-,]*(occipital|temporal|parietal)\s*((lobe|area)s?|(\.|\,|and|\s)*((?=left|right|[lL]t|[rR]t)|(?!=fronto)))'
                  , ' right-temporal-lobe right-parietal-lobe right-occipital-lobe left-temporal-lobe left-parietal-lobe left-occipital-lobe '
                  , text)
    #  both, bilateral 'frontal' + 'parietal' + 'temporal'
    text = re.sub(r'[Bb]oth\s*[PTF]\-[PTF]\-[PTF]\s*((lobe|area)(s)?)?|'
                  r'(at )?the.*?([Bb]ilateral|[Bb]oth|양측).*frontal parietal temporal lobe(s)?|'
                  r'(at )?(the )?([Bb]ilateral|[Bb]oth|양측) (temporal|parietal|frontal)[ &,]*(temporal|parietal|frontal)(\,|\s|\&|and|\-|및)*(temporal|parietal|frontal)\s*(lobe(s)?|(\.|\,|and|\s)*(?=left|right|[lL]t|[rR]t))|'
                  r'(at )?(the)?([Bb]ilateral|[Bb]oth|양측) (fronto|parieto|temporo)[ ,\-]*(fronto|parieto|temporo)[ ,\-]*(frontal|parietal|temporal)\s*((lobe|area)s?|(\.|\,|and|\s)*((?=left|right|[lL]t|[rR]t)|(?!=occipit)))'
                  , ' left-temporal-lobe left-parietal-lobe left-frontal-lobe right-temporal-lobe right-parietal-lobe right-frontal-lobe '
                  , text)
    #  both, bilateral 'frontal' + 'parietal' + 'occipital'
    text = re.sub(r'[Bb]oth\s*[POF]\-[POF]\-[POF]\s*(lobe(s)?)?|'
                  r'(at )?(the )?([Bb]ilateral|[Bb]oth|양측) (occipital|parietal|frontal)[ &,]*(occipital|parietal|frontal)(\,|\s|\&|and|\-)*(occipital|parietal|frontal)\s*(lobe(s)?|(\.|\,|and|\s)*(?=left|right))|'
                  r'(at )?(the )?([Bb]ilateral|[Bb]oth|양측) (fronto|parieto|occipito)[ ,\-&]*(fronto|parieto|occipito)[ ,\-&]*(occipital|parietal|frontal)\s*((lobe|area)s?|(\.|\,|and|\s)*((?=left|right|[lL]t|[rR]t)|(?!=tempor)))'
                  , ' left-occipital-lobe left-parietal-lobe left-frontal-lobe right-occipital-lobe right-parietal-lobe right-frontal-lobe '
                  , text)
    #  both, bilateral 'frontal' + 'temporal' + 'occipital'
    text = re.sub(r'[Bb]oth\s*[TOF]\-[TOF]\-[TOF]\s*(lobe(s)?)?|'
                  r'(at )?(the )?([Bb]ilateral|[Bb]oth|양측) (occipital|temporal|frontal)[ &,]*(occipital|temporal|frontal)(\,|\s|\&|and|\-)*(occipital|temporal|frontal)\s*(lobe(s)?|(\.|\,|and|\s)*(?=left|right))|'
                  r'(at )?(the )?([Bb]ilateral|[Bb]oth|양측) (fronto|temporo|occipito)[ &,\-]*(fronto|temporo|occipito)[ &,\-]*(frontal|temporal|occipital)\s*((lobe|area)s?|(\.|\,|and|\s)*((?=left|right|[lL]t|[rR]t)|(?!=pariet)))'
                  , ' left-occipital-lobe left-temporal-lobe left-frontal-lobe right-occipital-lobe right-temporal-lobe right-frontal-lobe '
                  , text)

    #  both 'parietal' + 'temporal'
    text = re.sub(r'[Bb]oth\s*[PT]\-[PT]\s*(lobe(s)?)?|both parieto-temporo-parietal lobe(s)?|'
                  r'(at )?(the )?([Bb]ilateral|[Bb]oth|양측) (temporal|parietal)(\.|\,|and|\s|\&)*(temporal|parietal)\s*((area|lobe)s?|(\.|\,|and|\s)*((?=left|right)|(?!=occipit|front)))|'
                  r'(at )?(the )?([Bb]ilateral|[bB]oth|양측) (parieto|temporo)[\- ,]*(parietal|temporal)\s*(lobe(s)?|(\.|\,|and|\s)*((?=left|right)|(?!=occipit|front)))'
                  , ' right-temporal-lobe right-parietal-lobe left-temporal-lobe left-parietal-lobe '
                  , text)
    #  both 'parietal' + 'occipital'
    text = re.sub(r'[Bb]oth\s*[PO]\-[PO]\s*(lobe(s)?)?|'
                  r'(at )?(the )?([Bb]ilateral|[Bb]oth|양측) (occipital|parietal|parieto|occipito)(\.|\,|and|\s|\&|\-)*(occipital|parietal)\s*((area|lobe)s?|(\.|\,|and|\s)*((?=left|right)|(?!=tempor|front)))'
                  , ' right-occipital-lobe right-parietal-lobe left-occipital-lobe left-parietal-lobe '
                  , text)
    #  both 'parietal' + 'frontal'
    text = re.sub(r'([Bb]oth|the [Bb]ilateral)\s*[PF]\-[PF]\s*(lobe(s)?)?|'
                  r'(at )?([Tt]he )?([Bb]ilateral|[Bb]oth|양측) (frontal|parietal|fronto|parieto)(\.|\,|and|\s|\&|\-|lobe)*(frontal|parietal)\s*((area|lobe)s?|(\.|\,|and|\s)*((?=left|right)|(?!=tempor|occipit)))'
                  , ' right-parietal-lobe right-frontal-lobe left-parietal-lobe left-frontal-lobe '
                  , text)
    #  both 'temporal' + 'occipital'
    text = re.sub(r'[Bb]oth\s*[TO]\-[TO]\s*(lobe(s)?)?|'
                  r'(at )?(the )?([Bb]ilateral|[Bb]oth|양측) (tempor|occipit?)(al|o)?(\.|\,|and|\s|\&|\-)*(temporal|occipital)\s*((area|lobe)s?|(\.|\,|and|\s)*((?=left|right)|(?!=pariet|front)))'
                  , ' right-temporal-lobe right-occipital-lobe left-temporal-lobe left-occipital-lobe '
                  , text)
    #  both 'temporal' + 'frontal'
    text = re.sub(r'[Bb]oth\s*[TF]\-[TF]\s*(lobe(s)?)?|'
                  r'(at )?([Tt]he )?([Bb]ilateral|[Bb]oth|양측) (frontal|fronto|temporal|temporo)(\.|\,|and|\s|\-|\&)*(frontal|temporal)\s*((area|lobe)s?|(\.|\,|and|\s)*((?=[Ll]eft|[Rr]ight)|(?!=pariet|occipit)))'
                  , ' right-temporal-lobe right-frontal-lobe left-temporal-lobe left-frontal-lobe '
                  , text)
    #  both 'occipital' + 'frontal'
    text = re.sub(r'[Bb]oth\s*[OF]\-[OF]\s*(lobe(s)?)?|'
                  r'(at )?([Tt]he )?([Bb]ilateral|[Bb]oth|양측) (frontal|fronto|occipital|occipito)(\.|\,|and|\s|\-|\&)*(frontal|occipital)\s*((area|lobe)s?|(\.|\,|and|\s)*((?=[Ll]eft|[Rr]ight)|(?!=pariet|tempor)))|'
                  r'(at )?([Tt]he )?([Bb]ilateral|[Bb]oth|양측) (front|occipit)(al|o)( lobes?)?[ ,]*(front|occipit)(al|o)\s*lobes?'
                  , ' right-occipital-lobe right-frontal-lobe left-occipital-lobe left-frontal-lobe '
                  , text)

    #  both 'parietal'
    text = re.sub(r'[Bb]oth P[ ,)]lobe(s)?|'
                  r'(at )?(the )?([Bb]ilateral|[Bb]oth|양측)\s*par(i)?etal\s*((lobe|area)s?|(and|\,|\s|\&)*(?!=(front|occipit|tempor)))'
                  , ' right-parietal-lobe left-parietal-lobe '
                  , text)
    #  both 'temporal'
    text = re.sub(r'[Bb]oth T[ ,)]lobe(s)?|'
                  r'(at )?(the )?([Bb]ilater(r)?al|[Bb]oth|양측)\s*temporal\s*((lobe|area)s?|(and|\,|\s|\&)*(?!=(front|occipit|pariet)))'
                  , ' right-temporal-lobe left-temporal-lobe '
                  , text)
    #  both 'occipital'
    text = re.sub(r'[Bb]oth O[ ,)]lobe(s)?|'
                  r'(at )?(the )?([Bb]ilateral|[Bb]oth|양측)\s*occ(i)?p(i)?tal\s*((lobe|area)s?|(and|\,|\s|\&)*(?!=(front|tempor|pariet)))'
                  , ' right-occipital-lobe left-occipital-lobe '
                  , text)
    #  both 'frontal'
    text = re.sub(r'[Bb]oth F[ ,)]lobe(s)?|'
                  r'(at )?(the )?([Bb]ilateral|[Bb]oth|양측)\s*frontal\s*((lobe|area)s?|(and|\,|\s|\&)*(?!=(occipit|tempor|pariet)))'
                  , ' right-frontal-lobe left-frontal-lobe '
                  , text)

    #  both 'cerebellum' + 'cerebrum'
    text = re.sub(r'(at )?(the )?([Bb]ilateral|[Bb]oth|양측)\s*(cerebral|cerebellar)( |and|\&|\-)*(cerebral|cerebellar)'
                  , r' right-cerebellum right-cerebrum left-cerebellum left-cerebrum '
                  , text)
    #  both 'cerebral' or 'cerebrum'
    text = re.sub(r'(at )?(the )?([Bb]oth|[Bb]ilateral|bialteral|양측)\s*([Cc]erebral|[Cc]erebrum)'
                  , ' right-cerebrum left-cerebrum '
                  , text)
    #  both 'cerebellum'
    text = re.sub(r'(at )?(the )?([Bb]oth|[Bb]ilateral|bialteral|양측)\s*[Cc]erebellum'
                  , ' right-cerebellum left-cerebellum '
                  , text)



    #  right 4개 항목
    text = re.sub(r'(at )?(the )?([rR]ight|[rR]t\.?) (frontal|parietal|temporal|occipital)[ ,]*(frontal|parietal|temporal|occipital)[ ,]*(frontal|parietal|temporal|occipital)( |\,|and)*(frontal|parietal|temporal|occipital)[ ,]*(lobe|area)s?'
                  , r' right-temporal-lobe right-parietal-lobe right-occipital-lobe right-frontal-lobe '
                  , text)

    #  right 'parietal' + 'temporal' + 'occipital'
    text = re.sub(r'(at )?(the )?([rR]ight|[rR]t)\.?\s*[PTO]\-[PTO]\-[PTO]\s*((lobe|area)(s)?)?|'
                  r'(at )?(the )?([Rr]ight|[rR]t\.?) (pariet|tempor|occipit)(al|o)[, &\-]*(pariet|tempor|occipit)(al|o)(\,|\s|\&|and|\-)*(pariet|tempor|occ(i)?pit)(al|o)(\,|\s|\&|and|\-)*((lobe|area)s?\.?|(?=left|both)|(?!=front))|'
                  r'(at )?(the )?([Rr]ight|[rR]t\.?) [PTO]+\(((pariet|tempor|occipit)(al|o)|\s|\-)+\)\s*lobes?\.?'
                  , ' right-temporal-lobe right-parietal-lobe right-occipital-lobe '
                  , text)
    #  right 'parietal' + 'temporal' + 'frontal'
    text = re.sub(r'(at )?(the )?([rR]ight|[rR]t)\.?\s*[PTF]\-[PTF]\-[PTF]\s*((lobe|area)(s)?)?|'
                  r'(at )?(the )?([Rr]ight|[rR]t\.?) (pariet|tempor|fron(t)?)(al|o)[, &\-]*(pariet|tempor|front)(al|o)(\,|\s|\&|and|\-)*(pariet|tempor|front)(al|o)(\,|\s|\&|and|\-)*((lobe|area)s?\.?|(?=left|both)|(?!=occipit))'
                  , ' right-temporal-lobe right-parietal-lobe right-frontal-lobe '
                  , text)
    #  right 'parietal' + 'occipital' + 'frontal'
    text = re.sub(r'(at )?(the )?([rR]ight|[rR]t)\.?\s*[POF]\-[POF]\-[POF]\s*((lobe|area)(s)?)?|'
                  r'(at )?(the )?([Rr]ight|[rR]t\.?) (pariet|occipit|front)(al|o)[, &\-]*(pariet|occipit|front)(al|o)(\,|\s|\&|and|\-)*(pariet|occ(i)?pit|front)(al|o)(\,|\s|\&|and|\-)*((lobe|area)s?\.?|(?=left|both)|(?!=tempor))'
                  , ' right-occipital-lobe right-parietal-lobe right-frontal-lobe '
                  , text)
    #  right 'temporal' + 'occipital' + 'frontal'
    text = re.sub(r'(at )?(the )?([rR]ight|[rR]t)\.?\s*[TOF]\-[TOF]\-[TOF]\s*((lobe|area)(s)?)?|'
                  r'(at )?(the )?([Rr]ight|[rR]t\.?) (tempor|occipit|front)(al|o)[, &\-]*(tempor|occipit|front)(al|o)(\,|\s|\&|and|\-)*(tempor|occ(i)?pit|front)(al|o)(\,|\s|\&|and|\-)*((lobe|area)s?\.?|(?=left|both)|(?!=pariet))'
                  , ' right-occipital-lobe right-temporal-lobe right-frontal-lobe '
                  , text)

    #  right 'parietal' + 'temporal'
    text = re.sub(r'(at )?(the )?([Rr](i)?ght|[rR]t)\.?\s*[PT]\-[PT]\s*((lobe|area)(s)?)?|'
                  r'(at )?(the )?([Rr]ight|[rR]t\.?) (pariet|tempor)(al|o)(\,|\s|\&|and|\-|lobe)*(pariet|tempor)(al|o)(\,|\s|\&|and|\-)*((lobe|area)s?\.?|(?=left|both)|(?!=occipit|front))'
                  , ' right-temporal-lobe right-parietal-lobe '
                  , text)
    #  right 'parietal' + 'occipital'
    text = re.sub(r'(at )?(the )?(right|[rR]t)\.?\s*[PO]\-[PO]\s*((lobe|area)(s)?)?|'
                  r'(at )?(the )?([Rr]ight|[rR]t\.?) (pariet|occipit)(al|o)(\,|\s|\&|and|\-)*(pariet|occipit)(al|o)(\,|\s|\&|and|\-)*((lobe|area)s?\.?|(?=left|both)|(?!=tempor|front))'
                  , ' right-occipital-lobe right-parietal-lobe '
                  , text)
    #  right 'parietal' + 'frontal'
    text = re.sub(r'(at )?(the )?(right|[rR]t)\.?\s*[FP](\-|\, )[FP]\s*((lobe|area)(s)?)?|'
                  r'(at )?(the )?([rR]ight|[rR]t\.?) (front|pariet)(al|o)(\,|\s|\&|and|\-|lobe)*(front|pariet)(al|o)(\,|\s|\&|and|\-)*((lobe|area)s?\.?|(?=left|both)|(?!=tempor|occipit))'
                  , ' right-frontal-lobe right-parietal-lobe '
                  , text)
    #  right 'temporal' + 'occipital'
    text = re.sub(r'(at )?(the )?(right|[rR]t)\.?\s*[TO]\-[TO]\s*((lobe|area)(s)?)?|'
                  r'(at )?(the )?([rR]ight|[rR]t\.?) (tempor|occipit)(al|o)(\,|\s|\&|and|\-)*(tempor|occipit)(al|o)(\,|\s|\&|and|\-)*((lobe|area)s?\.?|(?=left|both)|(?!=front|pariet))'
                  , ' right-temporal-lobe right-occipital-lobe '
                  , text)
    #  right 'temporal' + 'frontal'
    text = re.sub(r'(at )?(the )?(right|[rR]t)\.?\s*[FT]\-[FT]\s*((lobe|area)(s)?)?|'
                  r'(at )?(the )?([rR]ight|[rR]t\.?) (tempor|f(ro|or)nt)(al|o)(\,|\s|\&|and|\-)*(tempor|front|temopr)(al|o)(\,|\s|\&|and|\-)*((lobe|area)s?\.?|(?=left|both)|(?!=occipit|pariet))'
                  , ' right-temporal-lobe right-frontal-lobe '
                  , text)
    #  right 'occipital' + 'frontal'
    text = re.sub(r'(at )?(the )?(right|[rR]t)\.?\s*[OF]\-[OF]\s*((lobe|area)(s)?)?|'
                  r'(at )?(the )?([Rr]ight|[rR]t\.?) (occipit|front)(al|o)(\,|\s|\&|and|\-)*(occipit|front)(al|o)(\,|\s|\&|and|\-)*((lobe|area)s?\.?|(?=left|both)|(?!=tempor|pariet))'
                  , ' right-occipital-lobe right-frontal-lobe '
                  , text)

    #  right 'parietal'
    text = re.sub(r'(at )?(the )?([rR][Tt]\.?|[Rr]ight) P[ ,).]|'
                  r'(at )?(the )?([Rr]ight|[rR][tT]\.?) (pari(e)?t|paret)(al?|o)(\,|\s|\&|and|\-)*((lobe|area)(s|에)?\.?|(?=left|both)|(?!=(occipit|front|tempor)))|'
                  r'우측 두정엽'
                  , ' right-parietal-lobe '
                  , text)
    #  right 'temporal'
    text = re.sub(r'([rR][Tt]\.?|[Rr]ight) T[ ,).]|'
                  r'(at )?(the )?([Rr]ight|[rR][tT]\.?) tempor(al|o)(\,|\s|\&|and|\-)*((lobe|area)(s|에)?[.-]*|(?=left|both)|(?!=(occipit|front|pariet)))|'
                  r'우측 측두엽'
                  , ' right-temporal-lobe '
                  , text)
    #  right 'occipital'
    text = re.sub(r'([rR][Tt]\.?|[Rr]ight) O[ ,).]|'
                  r'(at )?(the )?([Rr]ight|[rR][tT]\.?) (occ(i)?pit|occipti)(al|o)(\,|\s|\&|and|\-)*((lobe|area)(s|에)?\.?|(?=left|both)|(?!=(pariet|front|tempor)))|'
                  r'우측 후두엽'
                  , ' right-occipital-lobe '
                  , text)
    #  right 'frontal'
    text = re.sub(r'([rR][Tt]\.?|[Rr]ight) F[ ,).]|'
                  r'(at )?(the )?([Rr]ig(ht|th)|[rR][tT]\.?) front(al|o)(\,|\s|\&|and|\-)*((lobe|area)(s|에)?\.?|(?=[Ll]eft|[lL]t|both)|(?!=(pariet|occipit|tempor)))|'
                  r'우측 전두엽'
                  , ' right-frontal-lobe '
                  , text)
    #  right 'cerebellum'
    text = re.sub(r'(at )?(the )?([rR][Tt]\.?|[Rr]ight) ([cC][Bb][lL][lL]|cerebellum|cerebellar)\.?'
                  , ' right-cerebellum '
                  , text)
    #  right 'cerebrum'
    text = re.sub(r'(at )?(the )?([rR][Tt]\.?|[Rr]ight) (cer(e)?bral|cerebrum)\.?'
                  , ' right-cerebrum '
                  , text)


    #  left 4개 항목
    text = re.sub(r'(at )?(the )?([lL]eft|[lL]t\.?) (front|pariet|tempor|occipit)(al|o)[ ,\-]*(front|pariet|tempor|occipit)(al|o)[ ,\-]*(front|pariet|tempor|occipit)(al|o)( |\,|and|\-)*(front|pariet|tempor|occipit)(al|o)[ ,]*(lobe|area)s?'
                  , r' left-temporal-lobe left-parietal-lobe left-occipital-lobe left-frontal-lobe left-cerebellum '
                  , text)

    #  left 'parietal' + 'temporal' + 'ocipital'
    text = re.sub(r'([Ll]eft|[lL]t)\.?\s*[PTO]\-[PTO]\-[PTO]\s*(lobe(s)?)?|'
                  r'(at )?(the )?([lL]eft|[lL]t\.?) (pariet|tempor|occipit)(al|o)[, &\-]*(pariet|tempor|occipit)(al|o)(\,|\s|\&|and|\-)*(pariet|tempor|occipit)(al|o)(\,|\s|\&|and|\-)*((lobe|area)s?\.?|(?=[Rr]ight|both)|(?!=front))'
                  , ' left-temporal-lobe left-parietal-lobe left-occipital-lobe '
                  , text)
    #  left 'parietal' + 'temporal' + 'frontal'
    text = re.sub(r'([Ll]eft|[lL]t)\.?\s*[PTF]\-[PTF]\-[PTF]\s*(lobe(s)?)?|'
                  r'(at )?(the )?([lL]eft|[lL]t\.?) (pariet|tempor|front|fornt)(al|o)(\,|\s|\&|and|\-)*(pariet|tempor|front)(al|o)(\,|\s|\&|and|\-)*(pariet|tempor|front)(al|o)(\,|\s|\&|and|\-)*((lobe|area)s?\.?|(?=[Rr]ight|both)|(?!=occipit))'
                  , ' left-temporal-lobe left-parietal-lobe left-frontal-lobe '
                  , text)
    #  left 'parietal' + 'occipital' + 'frontal'
    text = re.sub(r'([Ll]eft|[lL]t)\.?\s*[POF]\-[POF]\-[POF]\s*(lobe(s)?)?|'
                  r'(at )?(the )?([lL]eft|[lL]t\.?) (pariet|occipit|front)(al|o)(\s*\(prefrontal gyrus\))?[, &\-]*(pariet|occipit|front)(al|o)(\,|\s|\&|and|\-)*(pariet|occ(i)?pit|front)(al|o)(\,|\s|\&|and|\-)*((lobe|area)s?\.?|(?=[Rr]ight|both)|(?!=tempor))'
                  , ' left-occipital-lobe left-parietal-lobe left-frontal-lobe '
                  , text)
    #  left 'temporal' + 'occipital' + 'frontal'
    text = re.sub(r'([Ll]eft|[lL]t)\.?\s*[TOF]\-[TOF]\-[TOF]\s*(lobe(s)?)?|'
                  r'(at )?(the )?([lL]eft|[lL]t\.?) (tempor|occipit|front)(al|o)[, &\-]*(tempor|occipit|front)(al|o)(\,|\s|\&|and|\-)*(tempor|occipit|front)(al|o)(\,|\s|\&|and|\-)*((lobe|area)s?\.?|(?=[Rr]ight|both)|(?!=pariet))'
                  , ' left-temporal-lobe left-occipital-lobe left-frontal-lobe '
                  , text)

    #  left 'parietal' + 'temporal'
    text = re.sub(r'([Ll]eft|[lL]t)\.?\s*[PT]\-[PT]\s*((lobe|area)(s)?)?|'
                  r'(at )?(the )?([Ll]eft|[lL]t\.?) (pariet|tempor)(al|o)( |and|\,|\&|\-|lobe)*(pariet|tempor)(al|o)(\,|\s|\&|and|\-)*((lobe|area)s?\.?|(?=[rR]ight|both)|(?!=front|occipit))'
                  , ' left-temporal-lobe left-parietal-lobe '
                  , text)
    #  left 'parietal' + 'occipital'
    text = re.sub(r'([Ll]eft|[lL]t)\.?\s*[PO]\-[PO]\s*((lobe|area)(s)?)?|'
                  r'(at )?(the )?([lL]eft|[lL]t\.?) (pari(et|te)|occipit)(al|o)( |\&|\,|and|\-|lobe)*(pariet|occipit)(al|o)(\,|\s|\&|and|\-)*((lobe|area)s?\.?|(?=[rR]ight|both)|(?!=front|tempor))'
                  , ' left-occipital-lobe left-parietal-lobe '
                  , text)
    #  left 'parietal' + 'frontal'
    text = re.sub(r'([Ll]eft|[lL]t)\.?\s*[PF]\-[PF]\s*((lobe|area)(s)?)?|'
                  r'the left.*?F, P lobes|'
                  r'(at )?(the )?([lL]eft|[lL]t\.?) (front|pariet)(al|o)( |\&|\,|and|\-|lobe)*(front|pariet|paro)(al|o)(\,|\s|\&|and|\-)*((lobe|area)s?\.?|(?=[rR]ight|both)|(?!=occipit|tempor))'
                  , ' left-frontal-lobe left-parietal-lobe '
                  , text)
    #  left 'temporal' + 'occipital'
    text = re.sub(r'([Ll]eft|[lL]t)\.?\s*[TO]\-[TO]\s*((lobe|area)(s)?)?|'
                  r'(at )?(the )?([lL]eft|[lL]t\.?)\s*(occipit|tempor)(al|o)( |\&|\,|and|\-|lobe)*(occipit|tempor)(al|o)(\,|\s|\&|and|\-)*((lobe|area)s?\.?|(?=[rR]ight|both)|(?!=front|pariet))'
                  , ' left-temporal-lobe left-occipital-lobe '
                  , text)
    #  left 'temporal' + 'frontal'
    text = re.sub(r'([Ll]eft|[lL]t)\.?\s*[FT]\-[FT]\s*((lobe|area)(s)?)?|'
                  r'(at )?(the )?([lL]eft|[lL]t\.?) (front|tempor)(al|o)(\s|\&|\,|and|\-|lobe)*(front|tempor)(al|o)(\,|\s|\&|and|\-)*((lobe|area)s?\.?|(?=[rR]ight|both)|(?!=occipit|pariet))'
                  , ' left-temporal-lobe left-frontal-lobe '
                  , text)
    #  left 'occipital' + 'frontal'
    text = re.sub(r'([Ll]eft|[lL]t)\.?\s*[OF]\-[OF]\s*(lobe(s)?)?|'
                  r'(at )?(the )?([lL]eft|[lL]t\.?) (occipit|front)(al|o)(\s|\&|\,|and|\-|lobe)*(occipit|front)(al|o)(\,|\s|\&|and|\-)*((lobe|area)s?\.?|(?=[rR]ight|both)|(?!=pariet|tempor))'
                  , ' left-occipital-lobe left-frontal-lobe '
                  , text)

    #  left 'parietal'
    text = re.sub(r'([lL][Tt]\.?|[Ll]eft) P[ ,)]|'
                  r'(at )?(the )?([Ll]eft|[lL][Tt]\.?|elft) (p(a)?riet|parite)(al|o)(\,|\s|\&|and|\-)*((lobe|area)s?\.?|(?=[rR]ight|both)|(?!=(front|occipit|tempor)))|'
                  r'좌측 두정엽'
                  , ' left-parietal-lobe '
                  , text)
    #  left 'temporal'
    text = re.sub(r'([lL][Tt]\.?|[lL]eft) T[ ,)]|'
                  r'(at )?(the )?([Ll]eft|[lL][Tt]\.?)\s*(tempo[tr]|tempror)(al|o)(\,|\s|\&|and|\-|\.)*((lobe|area)s?\.?|(?=[rR]ight|both)|(?!=(front|occipit|pariet)))|'
                  r'좌측 측두엽'
                  , ' left-temporal-lobe '
                  , text)
    #  left 'occipital'
    text = re.sub(r'([lL][Tt]\.?|[Ll]eft) O[ ,)]|'
                  r'(at )?(the )?([lL]eft|[lL][Tt]\.?) (occ(i)?pit|occip(i)?ti)(al|o)(\,|\s|\&|and|\-)*((lobe|area)s?\.?|(?=[rR]ight|both)|(?!=(front|pariet|tempor)))|'
                  r'좌측 후두엽'
                  , ' left-occipital-lobe '
                  , text)
    #  left 'frontal'
    text = re.sub(r'([lL][Tt]\.?|[Ll]eft) F[ ,):]|'
                  r'(at )?(the )?([Ll]e(f)?t|[lL][tT]\.?) front(al|o)(\,|\s|\&|and|\-)*((lobe|area|\,)s?\.?|(?=[rR]ight|both)|(?!=(occipit|pariet|tempor)))|'
                  r'좌측 전두엽'
                  , ' left-frontal-lobe '
                  , text)
    # left 'cerebellum'
    text = re.sub(r'(the )?([lL][Tt]\.?|[lL]eft) ([cC][Bb][lL][lL]|c(e)?rebellum|cerebellar)\.?'
                  , ' left-cerebellum '
                  , text)
    #  left 'cerebrum'
    text = re.sub(r'(the )?([lL][Tt]\.?|[lL]eft) (cerebral|cerebrum)\.?'
                  , ' left-cerebrum '
                  , text)

    #  기타 조합의 경우 별도 정형화 작업
    #  left 'parietal' + 'temporal' + 'frontal' + 'cerebellum'
    text = re.sub(r'left frontal parietal temporal lobes and cerebellum'
                  , r' left-frontal-lobe left-parietal-lobe left-temporal-lobe left-cerebellum '
                  , text)
    #  right 'temporal' + 'occipital' + 'cerebellum'
    text = re.sub(r'right temporooccipital lobe and cerebellum'
                  , r' right-occipital-lobe right-temporal-lobe right-cerebellum '
                  , text)
    #  both 'parietal' + 'temporal' + 'cerebellum'
    text = re.sub(r'both parietotemporal lobes cerebellum'
                  , r' left-parietal-lobe left-temporal-lobe left-cerebellum right-parietal-lobe right-temporal-lobe right-cerebellum '
                  , text)
    #  both 'parietal' + 'frontal' + 'cerebellum'
    text = re.sub(r'both frontoparietal lobes cerebellum'
                  , r' left-parietal-lobe left-frontal-lobe left-cerebellum right-parietal-lobe right-frontal-lobe right-cerebellum '
                  , text)
    # 'orbitofrontal'
    text = re.sub(r'([rR]ight|[lL]eft|[bB]oth)\s*orbitofrontal\s*(?:lobe|area)(s)?', r' \1-orbitofrontal-lobe ', text)

    return text


## 소견문에 포함된 의학 용어(약어 및 기호를 포함한 단어) 정형화 함수.
#  Findings, Conclusion에 작성된 의학 용어 정형화.
def medical_words_preprocessing(text : str) -> str :
    """
    Medical terminology normalization function for clinical findings.
    :param text: Medical impression string data (Findings, Conclusion).
    :return: Converted string data.
    """
    # "Axial T1WI, sagittal T1WI, axial T2WI, axial FLAIR, axial T2* GRE image 획득하였으며 조영증강을 시행함/시행하지 않음." 내용 제거.
    # 어떤 검사를 시행함 또는 시행하지 않음과 관련된 모든 항목을 제거 (검사를 시행하고 시행하지 않고는 뇌졸중 판단의 근거로 볼 수 없다.)
    text = re.sub(r'(Axial (T1WI|DWI)|Intracranial TOF MRA|Axial T2\*|Neck MRA)[, ]*(sagittal T1WI)?[, ]*(axial T2WI)?[, ]*(axial FLAIR)?[, ]*'
                         r'(axial T2\*)?[, ]*(GRE image)?[, ]*(axial DWI)?[, ]*(intracranial TOF MRA)?[, ]*'
                         r'(neck TOF MRA)?[, ]*.*조영증강[은을 ]+시행(함|하지 않았?음)\.?|'
                         r'White matter tract evaluation.*diffusion tensor imaging 시행(함|하지 않았?음)\.?|'
                         r'(MR brain venography|PC3D Brain MR venography).*시행(함|하지 않았?음)\.?'
                  ,' '
                  , text
                  , flags=re.IGNORECASE)

    text = re.sub(r'([nN]o\s)((?!at )\w+)(?:(?=\s|\.|\,|$))', r' \1-\2 ', text)                 # No + 용어를 1개의 토큰으로 만듦.
    text = re.sub(r'([sS]tage)\s*(\d+)[,. ]+', r' \1-\2 ', text)                                # stage 3,
    text = re.sub(r'\(CE\)', ' contrast-enhancement ', text)                                    # (CE)
    text = re.sub(r'\([nN]on CE\)', ' Non-contrast-enhancement ', text)                         # (Non CE)
    text = re.sub(r'\/c\s', r' with ', text)                                                    # /c = with
    text = re.sub(r'\b[cC] [sS]pine', r' c-spine ', text)                                       # c spine  = c-spine
    text = re.sub(r'\(IA\)\-?', ' stage-ia ', text)                                             # lung(IA)-NSCLC
    text = re.sub(r'[pP]\-[cC][Oo][Mm]\.?\s*a((\.|\))\)?|rtery)|'
                  r'anterior communicating artery'
                  , ' posterior-communicating-artery '
                  , text)                                                                                   # anterior communicating artery
    text = re.sub(r'[pP]\-[cC][Oo][Mm](\.|\&)?|[pP][cC][oO][mM](\.|\&)?|'
                  r'posterior communicating'
                  , ' posterior-communicating '
                  , text)                                                                                   # posterior communicating, PCOM
    text = re.sub(r'[aA]\-[cC][oO][mM]\.?|[aA][cC][oO][mM]\.?|'
                  r'anterior communicating', ' anterior-communicating ', text)                          # anterior communicating, ACOM
    text = re.sub(r'[Ll]arge [bB]-cell lymphoma', r'dlbcl', text)                               # large B-cell lymphoma, DLBCL
    text = re.sub(r'[nN]on[- ]*small cell lung cance(r)?', r'nsclc', text)                      # Non-small cell lung cancer.
    text = re.sub(r'[vV]on [Hh]ippel[- ][lL]indau [Dd]isease', r'vhl-disease', text)            # Von Hippel–Lindau disease.
    text = re.sub(r'([eE]vans|[Cc]allosal)\s*([iI]ndex|[aA]ngle)', r'\1-\2', text)              # 2개 혼합 용어
    text = re.sub(r'\bMM[.,]', r' multiple-myeloma ', text)                                     # MM을 길이 단위 mm으로 혼용되지 않도록.
    text = re.sub(r'[iI]ntracranial [tT][oO][fF] [mM][rR][aA]', 'intracranial-tof-mra', text)   # Intracranial TOF MRA
    text = re.sub(r'[Nn]eck [tT][oO][fF] [mM][rR][aA]', 'neck-tof-mra', text)                   # Neck TOF MRA
    text = re.sub(r'[Nn]eck [mM][rR][aA]', 'neck-mra', text)                                    # Neck MRA
    text = re.sub(r'감마나이프|[gG]amma[ \-][kK]nife', ' gammaknife ', text)                         # 감마나이프, Gamma knife
    text = re.sub(r'[cC]ircle of [wW]illis', 'circle-of-willis', text)
    text = re.sub(r'GRE [iI]mage', 'gre-image', text)
    text = re.sub(r'Clinical information\s*:|\*\s*CI\s?:|C\.?I[,: ;]+', '', text)               # Clinical information, CI:, CI,
    text = re.sub(r'[sS]\/[pP]', ' status-post ', text)                                         # s/p, S/P
    text = re.sub(r'[rR][/][oO]', ' rule-out ', text)                                           # r/o, R/O
    text = re.sub(r'[Oo]p\.\s*site[., (]|(at )?(the )?op site', ' operative-site ', text)       # op. site
    text = re.sub(r'[hH]\/[oOpP]|history of', r' history-of ', text)                            # h/o
    text = re.sub(r'[Ff][./-][Uu]|follow up|follow\-up|'
                  r'Fu (?=MR(I|A))', ' follow-up ', text)                                               # f/u, f-u, f.u
    text = re.sub(r'N\/V', r'nausea vomiting', text)                                            # N/V = Nausea and Vomiting
    text = re.sub(r'[tT]2\*', r' t2-star ', text)                                               # T2*
    text = re.sub(r'[Tt]2[/\-][fF][lL][aA][iI][rR]', r' t2-flair ', text)                       # T2/FLAIR
    text = re.sub(r'\b([Tt][12]) hyperintens(e|ities|ity)', r' \1-hyperintense ', text)         # t2,1 hyperintense
    text = re.sub(r'\b[wW][/-][uU]\.?\b', r'work-up', text)                                     # w/u, W/U.
    text = re.sub(r'jx\.', r' junction ', text)                                                 # jx.
    text = re.sub(r'[iI]nverted ([TVYU])', r' inverted-\1 ', text)                              # inverted T 등
    text = re.sub(r'(CN|[cC]ranial [nN]erve)\s*([IV1-3]+)'
                        , lambda m:
                        ' cn cn-ophthalmic' if m.group(2) == 'V1' else
                        ' cn cn-maxilary' if m.group(2) == 'V2' else
                        ' cn cn-mandibular' if m.group(2) == 'V3' else r' cn-v '
                        , text)                                                                             # CN V
    text = re.sub(r'(?<=LC)[ \(]*([ABC])[, ]*(?:CP)?[ :,]([0-9ABC]+)\)?', r' lc-grade-\1-\2 ',
                   text)                                                                                    # LC(B, CP:6A)   - Child-Pugh score
    text = re.sub(r'(?:[gG]rade|[gG]r\.)\s*(\d+|[iI]+)\s*\)?\.?', r' grade-\1 ', text)          # grade 2 등
    text = re.sub(r'[zZ]one (\d+)', r' Zone-\1 ', text)                                         # Zone 1
    text = re.sub(r'[cC][/][Ww]', ' consistent-with ', text)
    text = re.sub(r'[fF]\/[iI]', ' further-investigation ', text)
    text = re.sub(r'\b[Nn][./-][sS]|[nN]on?( other)? [sS]ignificant|without significant change|[nN]o evidence of significant|[nN]on\s*specific|비특이적'
                  , ' non-specific '
                  , text)                                                                                   # Non specific 관련
    text = re.sub(r'with or without', r'with-or-without', text)
    text = re.sub(r'\b(low|high) b value', r' \1-b-value ', text)                               # low or high b value
    text = re.sub(r'(\d+) b value(s)?', r' \1-cnt b-value ', text)                              # 3 b values
    text = re.sub(r'\s*(\-|\()?\s*[Dd][Dd][xX].?', ' ', text)                                   # (DDx. 구분 목적의 용어 삭제
    text = re.sub(r'[rR]ec\s*[\).]', ' ', text)                                                 # Rec)  구분 목적의 용어 삭제
    text = re.sub(r'[dD][/][tT]|due to', ' due-to ', text)                                      # d/t, due to
    text = re.sub(r'\([iI][dD][xX]\s*\d+.*?\)\.?', ' ', text)                                   # 영상 이미지의 인덱스와 관련된 설명은 전부 삭제.
    text = re.sub(r'imaging', 'image', text)                                                    # 'image' 통일.
    text = re.sub(r'A2[/\-]3', r' A2-segment A3-segment ', text)
    text = re.sub(r'\(?A(\d+)(\s|\.|에|가|\-|\,|\;|$|\)|s)(?:[sS]eg(e)?ment)?s?', r' A\1-segment ', text)     # A1, A2 등을 segment로 구분
    text = re.sub(r'P(\d+)(\s|\.|에|\-|\,|\)|$|s)(?:[sS]egment)?s?', r' P\1-segment ', text)     # P1, P2 등을 segment로 구분
    text = re.sub(r'P2\/3', r' P2-segment P3-segment ', text)
    text = re.sub(r'V3\/4', r' V3-segment V4-segment ', text)
    text = re.sub(r'M(\d+)(\s|\.|에|\-||\,|까|$|s)(?:[sS]eg(e)?ment)?s?', r' M\1-segment ', text)         # M1, M2 등을 segment로 구분
    text = re.sub(r'V(\d+)(\s|\.|에|\-|\,|의|s|$|\~|\)|\/)(?:[sS]egment)?s?', r' V\1-segment ', text)     # V1, V2 등을 segment로 구분
    text = re.sub(r'(type|Bipolar)\s*([IV]+)', r' \1-\2 ', text)                                # type IV, Bipolar I 등
    text = re.sub(r'\sC1(\s|$)', r' atlas ', text)
    text = re.sub(r'\sC2(\s|$)', r' axis ', text)
    text = re.sub(r'C1\,2', r' atlas-axis ', text)                                                  # C1 = Atlas, C2 = Axis
    text = re.sub(r'(ICA|ICH|PCA|SDH|SAH|EVD|VA|ACA|MCA|BG|CCA)s', r'\1', text)                     # 뒤에 복수형으로 붙는 약어들
    text = re.sub(r'([aA]xial|[Ss]agittal)\s*(T1WI|T2WI|FLAIR|t2-star|DWI)', r' \1-\2 ', text)      # Axial T1WI, sagittal T1WI, axial T2WI 등
    text = re.sub(r'op[. ]+bed[., (]|(at )?op bed\.|op[., ]*bed(에서)', ' operative-bed ', text)     # op.bed, op bed 등
    text = re.sub(r'[Pp]ost[ -]*op', r'postop', text)                                               # post op
    text = re.sub(r'중이-?(꼭지|유양)돌기염', r' otomastoiditis ', text)                             # 중이-꼭지돌기염, 유양돌기염
    text = re.sub(r'해면\-추체', r' cavernous-petrous ', text)                                       # 해면-추체
    text = re.sub(r'백질-회색질', r'white-and-gray-matter', text)                                    # 백질-회색질
    text = re.sub(r'백질|[wW]hite [mM]atter', 'white-matter', text)
    text = re.sub(r'회색질|[gG]ray [mM]atter', 'gray-matter', text)
    text = re.sub(r'큰 차이|큰차이|큰 변화', r'signific-diff', text)                                   # 큰 차이, 큰차이 용어 통일.
    text = re.sub(r'([a-zA-Z]+)\/([a-zA-Z]+)', r'\1-\2', text)                                      # toxic/metabolic 등 중간에 '/' 구분 문자 있는 용어 통일.
    text = re.sub(r'위얌감[가-힣]*', r'위약감', text)                                                  # 오타에 의해 [unk] 토큰 분류되는 단어 처리.
    text = re.sub(r'cerebelli\.', r'cerebellum', text)                                              # cerebellum 오타 정정.
    text = re.sub(r'씰룩\s*거림', r' lip-twitching', text)
    text = re.sub(r'시퀀스', r' sequence ', text)                                                    # 영어를 한글로 표기한 것 중 [unk] 토큰 분류되는 단어 처리.
    text = re.sub(r'\bight', r'right', text)                                                        # right 오타 = ight
    text = re.sub(r"beni'gn", r'benign', text)                                                      # benign 오타 수정
    text = re.sub(r'(lobe|post |mm)\-(?!\>)', r' \1 ', text)                                        # 특정 단어 뒤에 붙은 의미없는 '-' 기호 제거.
    text = re.sub(r'\-(insular|positive|T[21]\s|about|diffusion|well|sized|focal)', r' \1', text)   # 특정 단어 앞에 붙은 의미없는 '-' 기호 제거.
    text = re.sub(r'([RL]|os) (MCA|MRA)',
                   lambda m:
                   f'right {m.group(2)}' if m.group(1) == 'R' else
                   f'left {m.group(2)}' if m.group(1) == 'L' or m.group(1) == 'os' else ""
                   , text)                                                                                      # MCA, MRA 등의 부위(right, left) 표현 통일.
    text = re.sub(r'([0-9. \-]+)m\s', r' \1cm', text)                                                # "1.6-m" 등의 cm 오타 정정.
    return text


## 순서(1st, 2nd, 3rd - Ordinal) 및 수량(one, two, three - Cardinal) 표현 정형화 함수.
def cardi_ordinal_preprocessing(text : str) -> str :
    """
    Nomalizing cardinal(e.g. one, two, three) and ordinal(e.g. 1st, 2nd, 3rd) numbers in medical terminology.
    :param text: Medical impression string data (Findings, Conclusion).
    :return: Converted string data.
    """
    # 1-2th, 7&8th 등 2개 이상의 순서 표현이 혼합된 경우, 앞의 순서 데이터를 먼저 전처리.
    matches = re.findall(r'(\d+)[\-&]+(?=\d+\s*th)', text)
    if matches:
        for v in matches:
            if v == '1':
                text = re.sub(fr'({v})[\-&]+', r' one-st ', text)
            elif v == '2':
                text = re.sub(fr'({v})[\-&]+', r' two-nd ', text)
            elif v == '3':
                text = re.sub(fr'({v})[\-&]+', r' three-rd ', text)
            elif v == '4':
                text = re.sub(fr'({v})[\-&]+', r' four-th ', text)
            elif v == '5':
                text = re.sub(fr'({v})[\-&]+', r' five-th ', text)
            elif v == '6':
                text = re.sub(fr'({v})[\-&]+', r' six-th ', text)
            elif v == '7':
                text = re.sub(fr'({v})[\-&]+', r' seven-th ', text)
            elif v == '8':
                text = re.sub(fr'({v})[\-&]+', r' eight-th ', text)
            elif v == '9':
                text = re.sub(fr'({v})[\-&]+', r' nine-th ', text)

    # 숫자를 포함하여 부위의 번호(ex. 7th)를 표현하는 데이터 전처리.
    matches = re.findall(r'(\d*)(\-*)(\d+(?:th|st|nd|rd))', text)
    if matches:
        for grplist in matches:
            if grplist[0] == '' and grplist[1] == '' and grplist[2]:
                if grplist[2][0] == '1':
                    text = re.sub(r'1st', ' one-st ', text)
                elif grplist[2][0] == '2':
                    text = re.sub(r'2nd', ' two-nd ', text)
                elif grplist[2][0] == '3':
                    text = re.sub(r'3rd', ' three-rd ', text)
                elif grplist[2][0] == '4':
                    text = re.sub(r'4th', ' four-th ', text)
                elif grplist[2][0] == '5':
                    text = re.sub(r'5th', ' five-th ', text)
                elif grplist[2][0] == '6':
                    text = re.sub(r'6th', ' six-th ', text)
                elif grplist[2][0] == '7':
                    text = re.sub(r'7(th|(?=[ &8-9]+th))', ' seven-th ', text)
                elif grplist[2][0] == '8':
                    text = re.sub(r'8th', ' eight-th ', text)
                elif grplist[2][0] == '9':
                    text = re.sub(r'9th', ' nine-th ', text)
                elif 'th' in grplist[2]:
                    text = re.sub(r'\d+th', ' over-th ', text)

    # lesion의 개수를 전처리.
    matches = re.findall(r'\b(\d+) lesion', text)
    if matches:
        matches = list(set(matches))
        for grplist in matches:
            if grplist[0] == '1':
                text = re.sub(fr'{grplist[0]} lesion', r' one lesion ', text)
            elif grplist[0] == '2':
                text = re.sub(fr'{grplist[0]} lesion', r' two lesion ', text)
            elif grplist[0] == '3':
                text = re.sub(fr'{grplist[0]} lesion', r' three lesion ', text)
            elif grplist[0] == '4':
                text = re.sub(fr'{grplist[0]} lesion', r' four lesion ', text)
            elif grplist[0] == '5':
                text = re.sub(fr'{grplist[0]} lesion', r' five lesion ', text)
            elif grplist[0] == '6':
                text = re.sub(fr'{grplist[0]} lesion', r' six lesion ', text)
            elif grplist[0] == '7':
                text = re.sub(fr'{grplist[0]} lesion', r' seven lesion ', text)
            elif grplist[0] == '8':
                text = re.sub(fr'{grplist[0]} lesion', r' eight lesion ', text)
            elif grplist[0] == '9':
                text = re.sub(fr'{grplist[0]} lesion', r' nine lesion ', text)

    ## x. '개수'를 의미하는 수치 데이터 전처리.
    text = re.sub(r'(innumerable.*?)(\d+)\)', r'\1 \2-cnt ', text)                               # (innumerable, > 30)
    text = re.sub(r'(\d+)\s*개', r' \1-cnt ', text)                                              # 2개
    text = re.sub(r'(\d+)\s*\-?(vessel|patient|in number|faint|small|well\-defined|aneurysms?)'
                  , r' \1-cnt \2 ', text)                                                       # 숫자 + 단어

    return text



## 3, 2, 1차원 길이 표현 ( 2.5 x 1.5 x 0.5 cm | 20x15mm | 12mm 등 )에 대한 Length-Width-Height 정형화 함수
#  cm 단위는 mm 형식으로 맞춘다 (소수점 '.' 표현을 없애기 위한 목적)
#  3차원은 'Length + Width + Height',  2차원은 'Length + Width', 1차원은 'Length'로 표현.
def demention_preprocessing(text : str) -> str :
    """
    normalizing 3D, 2D and 1D numerical expression in medical terminology.
    :param text: Medical impression string data (Findings, Conclusion).
    :return: Converted string data.
    """
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
            text = re.sub(fr'([^1-9]|^|\(){Lvalue}(?=\s*(x|\*|X)\s*{Wvalue}\s*(x|\*|X)\s*{Hvalue}' + r'\s*\-{1,4}\>?)'
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
    matches = re.findall(r'(\d{1,2}(?:\.\d{1,2})?)\s*(?:cm|mm)?(x|\*|X)(?:\.)?\s*(\d{1,2}(?:\.\d{1,2})?)\s*(?:cm|mm)?(x|\*|X)(?:\.)?\s*(\d{1,2}(?:\.\d{1,2})?)\s*(?:\(neck\))?\s*(cm|mm| )'
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
                text = re.sub(fr'(?<=Length-{grplist[0]}mm ){grplist[2]}[ xXm*]*(?={grplist[4]}\s*(\(neck\))?[ m]*)'
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
    matches = re.findall(r'(\d{1,2}(?:\.\d{1,2})?)\s*(x|\*|X)\s*(\d{1,2}(?:\.\d{1,2})?)\s*(?:cm|mm)?\s*(\-{1,4}\>*)(\s*<)?(?!cm|mm)'
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
            text = re.sub(fr'([^1-9]|^|\(){Lvalue}' + r'(?=\s*(x|\*|X)\s*\d{1,2}(\.\d{1,2})?\s*(cm|mm)?\s*\-{1,4}\>*)(\s*<)?(?!cm|mm)'
                          , fr' Length-{Ltmp}mm '
                          , text)

            # Width 정형화
            text = re.sub(fr'(?<=Length\-{Ltmp}mm ).+{Wvalue}\s*(cm|mm)?' + r'\s*\-{1,4}\>*(\s*<)?'
                          , fr'Width-{Wtmp}mm change '
                          , text)
            # print(matches)
            # print(Ftext)

    # 2차원 크기 데이터 정형화
    matches = re.findall(r'(\d{1,2}(?:\.\d{1,2})?)\s*(?:cm|mm)?(x|\*|X|\&)(?:\.)?\s*(\d{1,2}(?:\.\d{1,2})?)\s*(?:\-)?(cm|mm|\))'
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
                text = re.sub(fr'([^1-9]|^){Lvalue}' + r'(?=\s*(cm)?(x|\*|X)\.?\s*(\d{1,2}(\.\d{1,2})?)\s*\-?(cm|\)))'
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
                    text = re.sub(fr'([^1-9]|^){Lvalue}' + r'(?=\s*(mm)?(x|\*|X)\.?\s*(\d{1,2}(\.\d{1,2})?)\s*\-?mm)'
                                  , fr' Length-{Ltmp}mm '
                                  , text)
                else:
                    # Length 정형화
                    text = re.sub(fr'([^1-9]|^){grplist[0]}' + r'(?=\s*(mm)?(x|\*|X|\&)\.?\s*(\d{1,2}(\.\d{1,2})?)\s*\-?mm)'
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
        converted = [(s.replace(',', '.'), unit) for s, unit in matches]                    # ','로 오타가 난 실수 데이터를 '.'로 변환.
        matches = sorted(list(set(converted)), key=lambda x: float(x[0]), reverse=True)     # 실수 값을 기준으로 내림차순 정렬 (의도되지 않은 중복 정제 오류 방지 목적).
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
                masked = re.sub(fr'((?<!\d)|^|\(|(?<!Length\-)){grplist[0]}\s*mm[가-힣]*', fr' Length-{grplist[0]}mm ',
                                masked)

            # 마스킹된 텍스트를 복원 후, 최종 결과를 Ctext에 저장.
            for token, tmp in mask_matches:
                masked = masked.replace(token, tmp)

            text = masked
            masked = mask_pattern.sub(Mask_Repl, text)

    #  두 크기(길이) 값 사이에 존재하는 '변동'을 의미하는 특수 문자의 정형화 처리.
    text = re.sub(r'(?<=mm)\s*\-+\>\s*(?=Length\-)', ' change ', text)

    return text



## 날짜, 시간, 영상 인덱스 번호 등, 구분 주제(Note, DDX, e.g. 등) 문자열 전처리 함수.
def unnecessary_preprocessing(text : str) -> str :
    """
    Function to remove unnecessary string data
    :param text: Medical impression string data (Findings, Conclusion).
    :return: Converted string data.
    """
    # 날짜 기록 데이터 (2011.07.08.), (2011. 11. 11.), (2004) 전체 삭제.
    # 날짜 데이터는 '이전'의 의미를 전달할 뿐, 크게 의미 있지 않다고 판단하여 텍스트 삭제 진행.
    text = re.sub(r'\(?\d{2,4}[. ]+\d{1,2}[. ]+\d{1,2}[. ]+?\)?\\?|'    # (14.02.15.)
                  r'\(\d{4}\)|\(\d{1,2}\/\d{1,2}\)\.|'                         # (5/19).                          
                  r'\(?\d{4}([.\- ]*\d{1,2}[.\- ]*\d{1,2}[.\-), ]*)+|'         # (2020-09-09, 09-21)
                  r'on (\d+\/\d+\/\d{4}|\d{1,2}\/\d{1,2})\)?\.?|'              # on 5/8/2024, on 5/16
                  r'on \d{4}[\-.]\d{1,2}[\-.]\d{1,2}[).,]*|'                   # on 2021.5.28,  on 2022-01-05).
                  r'in \d{4}(\)\.)|'                                           # in 2024).
                  r'\d+년\s*\d{1,2}월\s*\d{1,2}일'                              # 2020년 6월 10일
                  r'밤 \d+시경|'                                                # 밤 11시경
                  r'[0-9 \-]+[0-9]일|'                                         # 6 - 9일
                  r'for[ 0-9]+days?|'                                         # for 2 days
                  r'in \d{4}[ .]+\d{1,2}[. ]+\d{1,2}[., ]+'                   # in 2017.9.13,
                  , ' ', text)

    # 특정 구분 텍스트 삭제 (DDX, Note 등)
    text = re.sub(r'\-\s*[iI][dD][xX][0-9 ,]*[iI][mM][0-9 ,]*[iI][dD][xX][0-9 ,]*[iI][mM][0-9 ,]*[.,\- ]?|'
                   r'[\[ ]*I[Dd][Xx][ 0-9]*I[Mm][ 0-9,-]*[\] ,]*|'
                   r'\([Ii][Dd][Xx][. 0-9]+image \d+\s*\)|'
                   r'\[stack.*?IDX[ 0-9-]*IM[ 0-9-*\]]*\.?|'
                   r'\([sS]e\d+[ ,]*I[Mm][ 0-9]*\)|'
                   r'\s[iI][Mm][ 0-9\).,\]]+|'
                   r'\(\#[iI][Dd][Xx][0-9 ,.\)\-\/]+|'
                   r'\(\d( |\,|and)*\d( |\,|and)*\d\)'
                   , ' ', text)                                                 # 영상 이미지 인덱스 번호 표현  ([IDX 4 IM 17].), (Se1, Im 15)

    # 'N년 전'의 의미를 갖는 문자열 전처리.
    text = re.sub(r'(\d+)\s*(yrs?|years?|months?)\s*ago'
                   , lambda m:
                   f' year-{m.group(1)}-be ' if m.group(2) in ['yr', 'yrs', 'year', 'years'] else
                   f' month-{m.group(1)}-be ' if m.group(2) in ['month', 'months'] else ""
                   , text)                                                                                  # 2 yr ago  = 2-year-before
    # 'after N-N years' 등의 범위로 작성된 문자열 전처리.
    text = re.sub(r'after\s*\d+[-~](\d+)\s*years?', r' year-\1-af ', text)                       # after 1~2 years
    # 'after N-N months' 등의 범위로 작성된 문자열 전처리.
    text = re.sub(r'after\s*\d+[-~](\d+)\s*months?', r' month-\1-af ', text)                     # after 6-12 months
    # 'after N-N weeks' 등의 범위로 작성된 문자열 전처리.
    text = re.sub(r'after\s*\d+[-~](\d+)\s*weeks?', r' week-\1-af ', text)                       # after 4-8 weeks
    # 'N년 이후에'의 의미를 갖는 문자열 전처리.
    text = re.sub(r'(?:after|\>)\s*(\d+)\s*(yrs?|years?|months?)\.?'
                   , lambda m:
                   f' year-{m.group(1)}-af ' if m.group(2) in ['yr', 'yrs', 'year', 'years'] else
                   f' month-{m.group(1)}-af ' if m.group(2) in ['month', 'months'] else ""
                   , text)                                                                                  # after 1 year  = 1-year-after

    text = re.sub(r'\*{2}[ 1-9\-,.]+', r' ', text)                                                            # ** 1-2, **1,2
    text = re.sub(r'\bI+\.\s', r' ', text)                                                                  # I., II., III.,
    text = re.sub(r'\*?\s*[nN]ote\s*[,:.]', r' ', text)                                                     # * Note:
    text = re.sub(r'(?<=[a-zA-Z가-힣\)])\.(\s|$|\n|\t)', r' ', text)                                         # 문장의 마지막 '.'
    text = re.sub(r'(\b|^)(\d|10)\s*[.,](\s|(?=MRA|Both|Right|Diff|Mic))|\d\s*(?=No -significant)'
                        , r' ', text)                                                                              # 1., 2., 3.,
    text = re.sub(r'[(\[]\d[)\]]\:?', r' ', text)                                                           # (1), (2), [1], [2] ...
    text = re.sub(r'(?<=\w)[`\'\’]([Ss]|\s)*', r' ', text)                                                  # Parkinson's
    text = re.sub(r'\s(\-+|\(|\))\s', r' ', text)                                                           # 구분 문자 역할의 ' - ' 등.
    text = re.sub(r'((\s|^|\*+)\d\))+[.,]?\s', r' ', text)                                                  # 순서 번호 역할의 1), 2).
    text = re.sub(r'e\.g\.?', r' ', text)                                                                   # e.g = for example
    text = re.sub(r'Ex\)', r' ', text)                                                                      # Ex)
    text = re.sub(r'박정식\.', r' ', text)                                                                   # 무의미한 사람 이름 포함.

    # 대소 비교 문자('<', '>') 텍스트 변환.
    # '<', '>'는 다른 특수 문자와 조합하여 '구분 문자' 역할로 사용하는 경우도 있다.
    # 이를 제외(삭제)하면 크기나 백분율 비교로 사용하므로 이들에 대한 명확한 텍스트 변환이 필요하다.
    # 대소 구분 목적으로 사용하는 기호 표현을 'less than', 'greater than'으로 텍스트 변경한다.
    text = re.sub(r'(\(|\b)?[lL]t\s', ' left ', text)
    text = re.sub(r'(\s|\()[rR]t[ )]', ' right ', text)
    text = re.sub(r'Lt\.?\s*\>\s*Rt\.?|Rt\.?\s*\<\s*Lt\.?|[lL]eft\s*\>\s*[rR]ight|[rR]ight\s*\<\s*[lL]eft'
                  , ' left greater-than right '
                  , text)
    text = re.sub(r'Rt\.?\s*\>\s*Lt\.?|Lt\.?\s*\<\s*Rt\.?|[rR]ight\s*\>\s*[lL]eft|[lL]eft\s*\<\s*[rR]ight'
                  , ' right greater-than left '
                  , text)
    text = re.sub(r'(?<!temporal)(?<!MRA)(?<!\-)(?<!ings)>\s*(?=[rR]ight|[Ll]eft|[rR]t|[lL]t|\d|[gG]rade|Length)|greater than'
                  , ' greater-than '
                  , text)
    text = re.sub(r'\(?<\s*(?=[rR]ight|[Ll]eft|[rR]t|[lL]t|\d|[gG]rade|Length)|less than'
                  , ' less-than '
                  , text)                                                                       # (< Length-5mm, (< Grade, ...
    text = re.sub(r'<(?=[a-zA-Z가-힣 ])|(?<=[a-zA-Z가-힣 *])>', ' ', text)                       # <Brain, dings>
    text = re.sub(r'(?<=[a-zA-Z가-힣])\s*[,:;]', ' ', text)                                      # DWI,
    # Ftext = re.sub(r'<', 'less than', Ftext)

    ## x. 구분자 역할의 특수문자 제거
    text = re.sub(r'\((?=[a-zA-Z])|(?<=[a-zA-Z])\)|'
                  r'[-=]+>|'
                  r'\(<-+|(?<=n) <- (?=s)|'
                  r'\*\.\s*\*\s*|'
                  r'\*+\d+[, ]\d*|'
                  r'\d', ' ', text)
    return text


## 양성(positive), 음성(negative) 내용 전처리.
def pos_neg_preprocessing(text : str) -> str:
    """
    Preprocessing of positive and negative contents.
    :param text: Medical impression string data (Findings, Conclusion).
    :return: Converted string data.
    """
    if '(+)' in text:
        text = re.sub(r'\(\+\)|\w\s\+', 'positive', text)

    if '(-)' in text:
        text = re.sub(r'\(\-\)', 'negative', text)

    if re.search(r'(MRA)\s*\:\s*\+', text):
        text = re.sub(r'(MRA)\s*\:\s*\+', r' \1 positive ', text)

    if re.search(r'(MRA)\s*\:\s*\-$', text):
        text = re.sub(r'(MRA)\s*\:\s*\-$', r' \1 negative ', text)

    return text


# Findings, Conclusion에서 특이적으로 발생한 전처리 작업을 하나로 통합.
def semi_preprocessing(text : str) -> str:
    tmp = text

    ##  1. 'ms'를 의미하는 수치 데이터 전처리.
    text = re.sub(r'\b(\d+)\s*(MS|ms)\b|TE\s+(\d+)'
                   , lambda m: (
            f"ms-{m.group(1)}" if m.group(1) else  # \1: 숫자 (ms)
            f"te ms-{m.group(3)}" if m.group(3) else ""  # TE 144 등의 ms없는 표현
        ), text)

    ##  2. % = percent 값 전처리.
    text = re.sub(r'(\d+)\s*\%', r' percent-\1 ', text)  # 10%  --> percent-10

    ##  3. Cho/NAA 수치 데이터 전처리.
    #   Cho/Cr = 2.29,  Cho/NAA = 1.14
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

    ##  4. 'Cho/Cr, Cho/NAA 수치의 기준 값 전처리.
    #   Cho/Cr = 2.29 (< 2.39). Cho/NAA = 1.14 (< 1.73).
    matches = re.findall(r'(Cho\-Cr|Cho\-NAA)\-(\d+\-\d+)\s*\(?([<>])\s*(\d+(?:\.\d+))\)?\.?', text)
    if matches:
        for grplist in matches:
            text = re.sub(fr'(?<={grplist[0]}-{grplist[1]})\s*\(?\<\s*', r' less-than ', text)
            text = re.sub(fr'(?<={grplist[0]}-{grplist[1]})\s*\(?\>\s*', r' greater-than ', text)
            tmp2 = grplist[3].replace('.', '-')
            text = re.sub(fr'({grplist[0]}-{grplist[1]}\s*(greater|less)-than)\s*{grplist[3]}\)?\.?',
                           fr' \1 {grplist[0]}-{tmp2} ', text)

    ##  5. 'ADC' 수치 데이터 전처리.
    #   ADC(avg) values of 0.862
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

    ##  6. Score 또는 검사 대상 수치 데이터 전처리
    #   ex) SIH score = 1/10, cistern 1/1, collection 0/1 등
    text = re.sub(r'(score|sinus|enhancement|collection|cistern(?:a)?|distance)\s*[ \=]*(\d+)\/(\d+)'
                   , r' \1 pos-\2 tot-\3 '
                   , text)  # 1/10   -->  'pos-1', 'tot-10'

    ##  7. aneurysm의 4차원 값 표현 구분
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
    sentences_list = []

    # Findings, Conclusion 데이터를 모두 사용하므로 하나의 구성요소로 만든다.
    for idx, Fs in enumerate(zip(DSet.Findings, DSet.Conclusion)):
        #Ftext = ' '.join(map(str, Fs[0].split('\n'))).strip()
        Ftext = Fs[0].replace('\r', ' ')
        #Ctext = ' '.join(map(str, Fs[1].split('\n'))).strip()
        Ctext = Fs[1].replace('\r', ' ')
        text = Ftext + ' ' + Ctext

        text = lobe_preprocessing(text)

        sentences = sent_tokenize(text)  # text를 tokenizing한 문장.
        Bert_sentences = ""
        for s in sentences:
            Bert_sentences += s + " __SEP__ "

        text = Bert_sentences

        text = semi_preprocessing(text)
        text = medical_words_preprocessing(text)
        text = cardi_ordinal_preprocessing(text)
        text = pos_neg_preprocessing(text)
        text = demention_preprocessing(text)
        text = unnecessary_preprocessing(text)

        text = '[CLS] ' + text
        text = re.sub(r'__SEP__', r'[SEP]', text)

        while True:
            new_text = re.sub(r'(?:\[(SEP|CLS)\])[ ]+\[SEP\]', r'[\1]', text)
            if new_text == text:
                break
            text = new_text

        # BERT에 적용할 Train Record 저장
        sentences_list.append(text)

    return sentences_list



# 전처리한 텍스트 데이터를 토큰화
def tokenizing(sentences: list, flags: int):
    global max_token_size

    # Training 시 토크나이저 사용 = 0
    # 검증 시 토크나이저 새로 불러오기 = 1
    if flags == 0 :
        global tokenizer_bert
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
    else :
        tokenizer_bert = AutoTokenizer.from_pretrained(model_save_path)

    # BERT Tokenizer 최대 길이 = 512
    MAX_LEN = 512
    tokenized_sentences = []

    # [CLS] [SEP] 또는 [SEP] [SEP]가 발생하는 경우를 제거.
    # 모든 텍스트를 소문자로 변환.
    for s in sentences:
        tokens = tokenizer_bert.tokenize(s)
        tokens = [tok if tok in ['[CLS]', '[SEP]', '[UNK]'] else tok.lower()
             for tok in tokens
             if (tok in ['[CLS]', '[SEP]', '[UNK]']) or (tok.lower() not in stopwords)]

        # if '[UNK]' in tokens:
        #     print(tokens)
        #     print(s)
        #     print('#################################')
        tokenized_sentences.append(tokens)

    # 단어 토큰에 고유한 인덱스 번호를 부여하고, 패딩을 첨가해 시퀀스 생성.
    input_ids = [tokenizer_bert.convert_tokens_to_ids(x) for x in tokenized_sentences]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")


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

    return input_ids, tokenized_sentences



def training(train_dataloader : DataLoader):
    device = Checking_cuda()
    gc.collect()
    torch.cuda.empty_cache()

    # BERT 학습 모델.
    config = BertConfig.from_pretrained(
        'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
        num_labels=2
    )

    model = BertForSequenceClassification.from_pretrained(
        'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
        config=config
    )

    # 모델을 학습 모드로 두고 진행.
    model.train()
    # 모델을 gpu에 담기.
    model.to(device)
    # 토크나이저 단어 사전에 사용자 추가된 것이 있으므로 개수 반영.
    model.resize_token_embeddings(len(tokenizer_bert))
    # 옵티마이저
    optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)

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




# Validation 데이터셋으로 검증 진행
def validation(test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gc.collect()
    torch.cuda.empty_cache()

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




## Sequence 데이터의 [PAD] 구분을 위한 Attention Mask 데이터 생성 함수.
# Parameter 'inputs'는 2차원 배열(numpy.ndarray)이므로 생성된 Attention mask도 2차원 배열 형태로 생성한다.
# [PAD]에 해당하는 '0'은 0으로 마스킹, 이외는 1로 마스킹
def attention_masking(inputs : list):
    """
    Mask the value '0' corresponding to [PAD] as 0, and all others as 1.
    :param inputs: A list of tokenized test(string).
    :return: Attention mask list.
    """
    att_list = []                                   # return Attention mask list
    for seq in inputs:
        seq_mask = [int(i > 0) for i in seq]      # [PAD] = 0, else = 1
        att_list.append(seq_mask)

    return att_list




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











## 불용어(and, the, 그, at 등) 제거 및 영문+한글('edema로', 'CTA를') 조합에서 불필요한 한글 제거.
def reorg_wordpieces(tokens : list):
    """
    Remove stopwords and preprocess words that combine English and Korean characters.
    :param tokens: A list of words split for tokenizaation.
    :return: A list of preprocessed words.
    """
    # 모든 토큰을 소문자 변환
    filtered_words = [tk.lower() for tk in tokens if tk.lower() not in stopwords]

    for idx, token in enumerate(filtered_words):
        # 변환된 토큰 중 BERT 구분자 [SEP]는 대문자로 변환 (대소문자를 구분하기 때문).
        if re.search(r'\[sep\]', token):
            token = '[SEP]'

        if re.search(r'[가-힣]', token) and re.search(r'[a-zA-Z-]', token):
            token = re.sub(r'[^a-zA-Z-]', '', token)                                # 영어 + 한글 조합의 토큰에서 한글 제거
            #filtered_words[idx] = token

        # 한글 표현에서 "[용어]의", "[용어]로" 등의 형태로 토큰화될 수 있는 표현 전처리
        if re.search(r'[가-힣]+([의에은을인이과]|으로)$', token):
            token = re.sub(r'([가-힣]+)(?:[의에은을인이과]|으로)$', r'\1', token)
        elif re.search(r'[a-zA-Z]+[로에은을이]$', token):
            token = re.sub(r'([a-zA-Z]+)[로에은을이]$', r'\1', token)

        # 동일한 의미를 여러 표현으로 나타내는 경우에 대한 정형화.
        token = re.sub(r'않(고|다|으며|는다|음|았음|아)$', r'않다', token)               # 않고, 않으며, 않음 등의 표현을 '않다'로 통일.
        token = re.sub(r'없(고|다|으며|음|었음|어|어보임)$', r'없다', token)             # 없고, 없으며, 없음 등의 표현을 '없다'로 통일.
        token = re.sub(r'있(고|다|으며|음|어|었떤)$', r'있다', token)                   # 있다, 있음 등의 표현을 '있다'로 통일.
        token = re.sub(r'(관찰|감소|증가|획득|발견|듯|환자).*', r'\1', token)           # 앞의 단어로 의미 설명이 충분한 것들.
        token = re.sub(r'보(임|이다|이며|인다|이는|이고|여)$', r'보이다', token)

        # 대/소문자 구분되는 특정 단어의 정형화
        if token == 'rt' or token == '우측':
            token = 'right'
        elif token == 'lt' or token == '좌측' or token =='촤측':
            token = 'left'
        elif re.search(r'소견[a-zA-Z가-힣]', token):
            token = '소견'
        elif token == '근위내경돔갱':         token = '근위내경동맥'          #오타 정정
        elif token == '임':                  token = '있다'               # 임 -> 이다 -> 있다
        elif token in ['분', '환자', 'pt']:   token = 'patient'           # 분 (시간 단위는 전처리해서 삭제되었음을 전제로 둠) -> 사람 -> 환자
        filtered_words[idx] = token

    return filtered_words







