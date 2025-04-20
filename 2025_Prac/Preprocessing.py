"""
- TrainCopySet.csv : 원본 데이터 Set
-
"""
import pandas as pd                         # DataFrame
import re                                   # Regular Expression
import nltk                                 # Word Tokenization
from nltk.tokenize import word_tokenize     # 단어 자연어 토큰화
from nltk.tokenize import sent_tokenize     # 문장 자연어 토큰화
from transformers import BertTokenizer
#nltk.download('punkt_tab')     # LookupError, punkt resource download
#from tensorflow.keras.preprocessing.sequence import pad_sequences  #Keras 시퀀스
# 토큰화 사전에 없는 용어 추가
vocab = ['찢어지는']
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
tokenizer_bert.add_tokens(vocab)

stopwords = ('and', 'at', 'a', 'an', 'as', 'are', 'in', 'to', 'the', 'of', 'or',
             'm', 's',
             '가', '과', '그', '그리고', '는', '년', '등의', '또는', '로', '및', '볼', '분', '수', '서'
             '에서', '이에', '이', '인', '의', '외', '와', '을', '인해', '에는', '에서', '에', '에도',
             '중', '지', '현', '함',
             '.', ',', ':', '(', ')', '→', '[', ']', '/', '*', '=', '+', "'", '&', '#')

def show_info(df: pd.DataFrame):
    """
    :param df: The DataFrame.
    :return: None. print 'info()'
    """
    print('----------------------------------------\n' \
          '-------@@@@ 원본 데이터 셋 정보 @@@@-------\n' \
          '----------------------------------------')
    df.info()
    print('-----------------------------------------------------------')

# Findings에는 1376개의 NaN(결측치) 데이터 존재.
# Conclusion에는 34개의 NaN(결측치) 데이터 존재.
# 결측값을 빈 문자열('')로 변환 처리.
def empty_to_missing(df : pd.DataFrame):
      print(f"Findings 결측값 = {df['Findings'].isnull().sum()}")
      print(f"Conclusion 결측값 = {df['Conclusion'].isnull().sum()}")

      # 모든 결측값에 빈 문자열 대체
      df.fillna('', inplace=True)
      print("@@@@ 결측값(NaN)을 빈 문자열('') 처리한다. @@@@\n -- 처리 결과 -- ")
      # 결측치 처리 결과
      print(f"Findings 결측값 처리 후 = {df['Findings'].isnull().sum()}")
      print(f"Conclusion 결측값 처리 후 = {df['Conclusion'].isnull().sum()}")
      print('-----------------------------------------------------------')


# 이미 정형화된 수치 데이터에 대한 마스킹 작업 (중복 변환 방지 목적).
mask_matches = []
def Mask_Repl(match) :
    token = f"__PROTECT{len(mask_matches)}__"
    mask_matches.append((token, match.group(0)))  # 원문 저장
    return token



## 대뇌의 4개(Parietal, Temporal, Occipital, Frontal)의 엽(Lobe) 분류 용어에 대한 정형화.
def lobe_preprocessing(text : str):
    #  both, bilateral 'parietal' + 'temporal' + 'occipital'
    text = re.sub(r'[Bb]oth\s*[PTO]\-[PTO]\-[PTO]\s*(lobe(s)?)?|'
                  r'(the )?([Bb]ilateral|[Bb]oth|양측) (temporal|parietal|occipital)[ &,]*(temporal|parietal|occipital)[ &,]*(temporal|parietal|occipital) lobe(s)?'
                   , ' right-temporal-lobe right-parietal-lobe right-occipital-lobe left-temporal-lobe left-parietal-lobe left-occipital-lobe '
                   , text)
    #  both, bilateral 'frontal' + 'parietal' + 'temporal'
    text = re.sub(r'[Bb]oth\s*[PTF]\-[PTF]\-[PTF]\s*(lobe(s)?)?|'
                   r'the.*?([Bb]ilateral|[Bb]oth|양측).*frontal parietal temporal lobe(s)?|'
                  r'(the )?([Bb]ilateral|[Bb]oth|양측) (temporal|parietal|frontal)[ &,]*(temporal|parietal|frontal)[ &,]*(temporal|parietal|frontal) lobe(s)?'
                   , ' left-temporal-lobe left-parietal-lobe left-frontal-lobe right-temporal-lobe right-parietal-lobe right-frontal-lobe '
                   , text)
    #  both, bilateral 'frontal' + 'parietal' + 'occipital'
    text = re.sub(r'[Bb]oth\s*[POF]\-[POF]\-[POF]\s*(lobe(s)?)?|'
                  r'(the )?([Bb]ilateral|[Bb]oth|양측) (occipital|parietal|frontal)[ &,]*(occipital|parietal|frontal)[ &,]*(occipital|parietal|frontal) lobe(s)?'
                   , ' left-occipital-lobe left-parietal-lobe left-frontal-lobe right-occipital-lobe right-parietal-lobe right-frontal-lobe '
                   , text)
    #  both, bilateral 'frontal' + 'temporal' + 'occipital'
    text = re.sub(r'[Bb]oth\s*[TOF]\-[TOF]\-[TOF]\s*(lobe(s)?)?|'
                  r'(the )?([Bb]ilateral|[Bb]oth|양측) (occipital|temporal|frontal)[ &,]*(occipital|temporal|frontal)[ &,]*(occipital|temporal|frontal) lobe(s)?'
                   , ' left-occipital-lobe left-temporal-lobe left-frontal-lobe right-occipital-lobe right-temporal-lobe right-frontal-lobe '
                   , text)

    #  both 'parietal' + 'temporal'
    text = re.sub(r'[Bb]oth\s*[PT]\-[PT]s*(lobe(s)?)?|both parieto-temporo-parietal lobe(s)?|'
                   r'(at )?(the )?([Bb]ilateral|[Bb]oth|양측) (temporal|parietal)[ &,]*(temporal|parietal)\s*(lobe(s)?|(\.|\,|and|\s)*(?=left|right))'
                   , ' right-temporal-lobe right-parietal-lobe left-temporal-lobe left-parietal-lobe '
                   , text)
    #  both 'parietal' + 'occipital'
    text = re.sub(r'[Bb]oth\s*[PO]\-[PO]s*(lobe(s)?)?|'
                   r'(at )?(the )?([Bb]ilateral|[Bb]oth|양측) (occipital|parietal)[ &,]*(occipital|parietal)\s*(lobe(s)?|(\.|\,|and|\s)*(?=left|right))|'
                   r'(at )?(the )?([Bb]ilateral|[bB]oth) parieto\-occipital\s*(lobe(s)?|(\.|\,|and|\s)*(?=left|right))'
                   , ' right-occipital-lobe right-parietal-lobe left-occipital-lobe left-parietal-lobe '
                   , text)
    #  both 'parietal' + 'frontal'
    text = re.sub(r'([Bb]oth|the [Bb]ilateral)\s*[PF]\-[PF]s*(lobe(s)?)?|'
                   r'(at )?(the )?([Bb]ilateral|[Bb]oth|양측) (frontal|parietal)[ &,]*(frontal|parietal)\s*(lobe(s)?|(\.|\,|and|\s)*(?=left|right))|'
                   r'(at )?([Tt]he )?([Bb]oth|bilateral) fronto(-)?parietal\s*(lobe(s)?)?((\.|\,|and|\s)*(?=[rR]ight|[lL]eft))?'
                   , ' right-parietal-lobe right-frontal-lobe left-parietal-lobe left-frontal-lobe '
                   , text)
    #  both 'temporal' + 'occipital'
    text = re.sub(r'[Bb]oth\s*[TO]\-[TO]s*(lobe(s)?)?|'
                   r'(the )?([Bb]ilateral|[Bb]oth|양측) (temporal|occipital)[ &,]*(temporal|occipital) lobe(s)?'
                   , ' right-temporal-lobe right-occipital-lobe left-temporal-lobe left-occipital-lobe '
                   , text)
    #  both 'temporal' + 'frontal'
    text = re.sub(r'[Bb]oth\s*[TF]\-[TF]s*(lobe(s)?)?|'
                   r'(the )?([Bb]ilateral|[Bb]oth|양측) (frontal|temporal)[ &,]*(frontal|temporal) lobe(s)?|'
                   r'(at )?([Tt]he )?[Bb]oth fronto-temporal lobe(s)?\s(and (?=[rR]ight|[lL]eft))?'
                   , ' right-temporal-lobe right-frontal-lobe left-temporal-lobe left-frontal-lobe '
                   , text)
    #  both 'occipital' + 'frontal'
    text = re.sub(r'[Bb]oth\s*[OF]\-[OF]s*(lobe(s)?)?|'
                   r'(the )?([Bb]ilateral|[Bb]oth|양측) (frontal|occipital)[ &,]*(frontal|occipital) lobe(s)?'
                   , ' right-occipital-lobe right-frontal-lobe left-occipital-lobe left-frontal-lobe '
                   , text)

    #  both 'parietal'
    text = re.sub(r'[Bb]oth P[ ,)]lobe(s)?|(the )?([Bb]ilateral|[Bb]oth) parietal lobe(s)?|'
                   r'(the )?([Bb]ilateral|[Bb]oth|양측) parietal[ ,&]\s*(?!=(frontal|occipital|temporal))'
                   , ' right-parietal-lobe left-parietal-lobe '
                   , text)
    #  both 'temporal'
    text = re.sub(r'[Bb]oth T[ ,)]lobe(s)?|(the )?([Bb]ilateral|[Bb]oth) temporal lobe(s)?|'
                   r'(the )?([Bb]ilateral|[Bb]oth|양측) temporal[ ,&]\s*(?!=(frontal|occipital|parietal))'
                   , ' right-temporal-lobe left-temporal-lobe '
                   , text)
    #  both 'occipital'
    text = re.sub(r'[Bb]oth O[ ,)]lobe(s)?|(the )?([Bb]ilateral|[Bb]oth) occipital lobe(s)?|'
                   r'(the )?([Bb]ilateral|[Bb]oth|양측) occipital[ ,&]\s*(?!=(frontal|parietal|temporal))'
                   , ' right-occipital-lobe left-occipital-lobe '
                   , text)
    #  both 'frontal'
    text = re.sub(r'[Bb]oth F[ ,)]lobe(s)?|(the )?([Bb]ilateral|[Bb]oth) frontal lobe(s)?|'
                   r'(the )?([Bb]ilateral|[Bb]oth|양측) frontal[ ,&]\s*(?!=(temporal|occipital|parietal))'
                   , ' right-frontal-lobe left-frontal-lobe '
                   , text)
    #  both 'cerebral' or 'cerebrum'
    text = re.sub(r'(the )?([Bb]oth|[Bb]ilateral|bialteral) ([Cc]erebral|[Cc]erebrum)', ' right-cerebrum left-cerebrum ', text)
    #  both 'cerebellum'
    text = re.sub(r'(the )?([Bb]oth|[Bb]ilateral|bialteral)\s*[Cc]erebellum', 'right-cerebellum left-cerebellum', text)

    #  right 'parietal' + 'temporal' + 'occipital'
    text = re.sub(r'(right|[rR]t)\.?\s*[PTO]\-[PTO]\-[PTO]\s*(lobe(s)?)?|'
                   r'(at )?(the )?[Rr]ight (parietal|temporal|occipital)[, &-]*(parietal|temporal|occipital)(\,|\s|\&|and|\-)*(parietal|temporal|occ(i)?pital) lobe(s)?'
                   , ' right-temporal-lobe right-parietal-lobe right-occipital-lobe '
                   , text)
    #  right 'parietal' + 'temporal' + 'frontal'
    text = re.sub(r'(right|[rR]t)\.?\s*[PTF]\-[PTF]\-[PTF]\s*(lobe(s)?)?|'
                   r'(at )?(the )?[Rr]ight (parietal|temporal|frontal)[, &]*(parietal|temporal|frontal)(\,|\s|\&|and)*(parietal|temporal|frontal) lobe(s)?'
                   , ' right-temporal-lobe right-parietal-lobe right-frontal-lobe '
                   , text)
    #  right 'parietal' + 'occipital' + 'frontal'
    text = re.sub(r'(right|[rR]t)\.?\s*[POF]\-[POF]\-[POF]\s*(lobe(s)?)?|'
                   r'(at )?(the )?[Rr]ight (parietal|occipital|frontal)[, &]*(parietal|occipital|frontal)(\,|\s|\&|and)*(parietal|occipital|frontal) lobe(s)?'
                   , ' right-occipital-lobe right-parietal-lobe right-frontal-lobe '
                   , text)
    #  right 'temporal' + 'occipital' + 'frontal'
    text = re.sub(r'(right|[rR]t)\.?\s*[TOF]\-[TOF]\-[TOF]\s*(lobe(s)?)?|'
                   r'(at )?(the )?[Rr]ight (temporal|occipital|frontal)[, &]*(temporal|occipital|frontal)(\,|\s|\&|and)*(temporal|occipital|frontal) lobe(s)?'
                   , ' right-occipital-lobe right-temporal-lobe right-frontal-lobe '
                   , text)

    #  right 'parietal' + 'temporal'
    text = re.sub(r'(r(i)?ght|[rR]t)\.?\s*[PT]\-[PT]s*(lobe(s)?)?|'
                   r'(at )?(the )?[Rr]ight (parietal|temporal)[ &,]*(and )?(parietal|temporal)\s*(lobe(s)?|(?!=\,?(occipital|frontal)))'
                   , ' right-temporal-lobe right-parietal-lobe '
                   , text)
    #  right 'parietal' + 'occipital'
    text = re.sub(r'(right|[rR]t)\.?\s*[PO]\-[PO]s*(lobe(s)?)?|'
                   r'(at )?(the )?[Rr]ight (parietal|occipital)[ &,]*(and )?(parietal|occipital)\s*(lobe(s)?|(?!=\,?(temporal|frontal)))|'
                   r'(at )?(the )?([Rr]ight|[rR]t\.?)\s*occipito-paretal\s*(area|lobe)s?'
                   , ' right-occipital-lobe right-parietal-lobe '
                   , text)
    #  right 'parietal' + 'frontal'
    text = re.sub(r'(right|[rR]t)\.?\s*[FP](\-|\, )[FP]s*(lobe(s)?)?|'
                   r'(at )?(the )?([rR]ight|[rR]t\.?) (frontal|parietal)[ &,\-]*(and )?(frontal|parietal)\s*(lobe(s)?|(?!=\,?(temporal|occipital)))'
                   , ' right-frontal-lobe right-parietal-lobe '
                   , text)
    #  right 'temporal' + 'occipital'
    text = re.sub(r'(right|[rR]t)\.?\s*[TO]\-[TO]s*(lobe(s)?)?|'
                   r'(at )?(the )?([rR]ight|[rR]t\.?) (temporal|occipital)[ &,]*(and )?(temporal|occipital)\s*(lobe(s)?|(?!=\,?(frontal|parietal)))'
                   , ' right-temporal-lobe right-occipital-lobe '
                   , text)
    #  right 'temporal' + 'frontal'
    text = re.sub(r'(right|[rR]t)\.?\s*[FT]\-[FT]s*(lobe(s)?)?|'
                   r'(at )?(the )?([rR]ight|[rR]t\.?) (temporal|frontal)[ &,]*(and )?(temporal|frontal)\s*(lobe(s)?|(?!=\,?(occipital|parietal)))'
                   , ' right-temporal-lobe right-frontal-lobe '
                   , text)
    #  right 'occipital' + 'frontal'
    text = re.sub(r'(right|[rR]t)\.?\s*[OF]\-[OF]s*(lobe(s)?)?|'
                   r'(at )?(the )?[Rr]ight (occipital|frontal)[ &,]*(and )?(occipital|frontal)\s*(lobe(s)?|(?!=\,?(temporal|parietal)))'
                   , ' right-occipital-lobe right-frontal-lobe '
                   , text)

    #  right 'parietal'
    text = re.sub(r'([rR][Tt]\.?|[Rr]ight) P[ ,)]|(at )?(the )?([Rr]ight|[rR][tT]\.?) parietal lobe(s|에)?\.?|'
                   r'(at )?(the )?([rR]ight|[rR]t\.?) parietal[ &,]*(?!=(occipital|frontal|temporal))'
                   , ' right-parietal-lobe '
                   , text)
    #  right 'temporal'
    text = re.sub(r'([rR][Tt]\.?|[Rr]ight) T[ ,)]|(at )?(the )?([Rr]ight|[rR][tT]\.?) temporal lobe(s|에)?\.?|'
                   r'(at )?(the )?([rR]ight|[rR]t\.?) temporal[ &,]*(?!=(occipital|parietal|frontal))'
                   , ' right-temporal-lobe '
                   , text)
    #  right 'occipital'
    text = re.sub(r'([rR][Tt]\.?|[Rr]ight) O[ ,)]|(at )?(the )?([Rr]ight|[rR][tT]\.?) occipital lobe(s|에)?\.?|'
                   r'(at )?(the )?([rR]ight|[rR]t\.?) occipital[ &,]*(?!=(parietal|frontal|temporal))'
                   , ' right-occipital-lobe '
                   , text)
    #  right 'frontal'
    text = re.sub(r'([rR][Tt]\.?|[Rr]ight) F[ ,)]|(at )?(the )?([Rr]ight|[rR][tT]\.?) frontal lobe(s|에)?\.?|'
                   r'(at )?(the )?([rR]ight|[rR]t\.?) frontal[ &,]*(?!=(occipital|parietal|temporal))|'
                   r'(at )?(the )?([rR]ight|[rR]t\.?) frontal(\s|\,|\&|and)*(?!=[lL]eft|[lL]t)'
                   , ' right-frontal-lobe '
                   , text)
    #  right 'cerebellum'
    text = re.sub(r'(the )?([rR][Tt]\.?|[Rr]ight) ([cC][Bb][lL][lL]|cerebellum|cerebellar)\.?', ' right-cerebellum ', text)
    #  right 'cerebrum'
    text = re.sub(r'(the )?([rR][Tt]\.?|[Rr]ight) (cerebral|cerebrum)\.?', ' right-cerebrum ', text)


    #  left 'parietal' + 'temporal' + 'ocipital'
    text = re.sub(r'(left|[lL]t)\.?\s*[PTO]\-[PTO]\-[PTO]\s*(lobe(s)?)?|'
                   r'(at )?(the )?([lL]eft|[lL]t\.?) (parietal|temporal|occipital)[, &]*(parietal|temporal|occipital)(\,|\s|\&|and)*(parietal|temporal|occipital) lobe(s)?'
                   , ' left-temporal-lobe left-parietal-lobe left-occipital-lobe '
                   , text)
    #  left 'parietal' + 'temporal' + 'frontal'
    text = re.sub(r'(left|[lL]t)\.?\s*[PTF]\-[PTF]\-[PTF]\s*(lobe(s)?)?|'
                   r'(at )?(the )?([lL]eft|[lL]t\.?) (parietal|temporal|frontal)[, &-]*(parietal|temporal|frontal)(\,|\s|\&|and|\-)*(parietal|temporal|frontal) lobe(s)?'
                   , ' left-temporal-lobe left-parietal-lobe left-frontal-lobe '
                   , text)
    #  left 'parietal' + 'occipital' + 'frontal'
    text = re.sub(r'(left|[lL]t)\.?\s*[POF]\-[POF]\-[POF]\s*(lobe(s)?)?|'
                   r'(at )?(the )?([lL]eft|[lL]t\.?) (parietal|occipital|frontal)(\s*\(prefrontal gyrus\))?[, &]*(parietal|occipital|frontal)(\,|\s|\&|and)*(parietal|occipital|frontal) lobe(s)?'
                   , ' left-occipital-lobe left-parietal-lobe left-frontal-lobe '
                   , text)
    #  left 'temporal' + 'occipital' + 'frontal'
    text = re.sub(r'(left|[lL]t)\.?\s*[TOF]\-[TOF]\-[TOF]\s*(lobe(s)?)?|'
                   r'(at )?(the )?([lL]eft|[lL]t\.?) (temporal|occipital|frontal)[, &]*(temporal|occipital|frontal)(\,|\s|\&|and)*(temporal|occipital|frontal) lobe(s)?'
                   , ' left-temporal-lobe left-occipital-lobe left-frontal-lobe '
                   , text)

    #  left 'parietal' + 'temporal'
    text = re.sub(r'(left|[lL]t)\.?\s*[PT]\-[PT]\s*(lobe(s)?)?|'
                   r'(at )?(the )?([Ll]eft|[lL]t\.?) (parietal|temporal)\s*(and|\,|\&)\s*(parietal|temporal) (lobe(s)?|(?!=(occipital|frontal)))|'
                   r'(at )?(the )?([Ll]eft|[lL]t\.?) (parieto|parietal)-temporal lobe(s)?\.?'
                   , ' left-temporal-lobe left-parietal-lobe '
                   , text)
    #  left 'parietal' + 'occipital'
    text = re.sub(r'(left|[lL]t)\.?\s*[PO]\-[PO]\s*(lobe(s)?)?|'
                   r'(at )?(the )?([lL]eft|[lL]t\.?) (parietal|occipital)[ &,]*(parietal|occipital) (lobe(s)?|(?!=(temporal|frontal)))|'
                   r'(at )?(the )?([lL]eft|[lL]t\.?) parieto(-)?occipital\s*(lobe(s)?)?'
                   , ' left-occipital-lobe left-parietal-lobe '
                   , text)
    #  left 'parietal' + 'frontal'
    text = re.sub(r'(left|[lL]t)\.?\s*[PF]\-[PF]\s*(lobe(s)?)?|'
                   r'the left.*?F, P lobes|'
                   r'(at )?(the )?([lL]eft|[lL]t\.?) (frontal|parietal)( |&|,|and|\-)*(frontal|parietal) (lobe(s)?|(?!=(temporal|occipital)))|'
                   r'(at )?(the )?([lL]eft|[lL]t\.?) fronto(-)?parietal\s*(lobe(s)?)?'
                   , ' left-frontal-lobe left-parietal-lobe '
                   , text)
    #  left 'temporal' + 'occipital'
    text = re.sub(r'(left|[lL]t)\.?\s*[TO]\-[TO]\s*(lobe(s)?)?|'
                   r'(at )?(the )?[lL]eft\s*(temporal|occipital|temporo)[ &,-]*(temporal|occipital)\s*(lobe(s)?|\,?(?!=(frontal|parietal)))'
                   , ' left-temporal-lobe left-occipital-lobe '
                   , text)
    #  left 'temporal' + 'frontal'
    text = re.sub(r'(left|[lL]t)\.?\s*[FT]\-[FT]\s*(lobe(s)?)?|'
                   r'(the )?[lL]eft (temporal|frontal)[ &,]*(temporal|frontal) (lobe(s)?|(?!=(occipital|parietal)))'
                   , ' left-temporal-lobe left-frontal-lobe '
                   , text)
    #  left 'occipital' + 'frontal'
    text = re.sub(r'(left|[lL]t)\.?\s*[OF]\-[OF]\s*(lobe(s)?)?|'
                   r'(at )?(the )?[lL]eft (occipital|frontal)[ &,]*(occipital|frontal) (lobe(s)?|(?!=(temporal|parietal)))'
                   , ' left-occipital-lobe left-frontal-lobe '
                   , text)

    #  left 'parietal'
    text = re.sub(r'([lL][Tt]\.?|[Ll]eft) P[ ,)]|(the )?(left|[lL][Tt]\.?) parietal\s*(lobe|and)|'
                   r'(the )?([lL]eft|[lL]t\.?) parietal[ &,]*(?!=(occipital|temporal|frontal))'
                   , ' left-parietal-lobe '
                   , text)
    #  left 'temporal'
    text = re.sub(r'([lL][Tt]\.?|[lL]eft) T[ ,)]|(the )?(left|[lL][Tt]\.?) temporal lobe(s)?\.?|'
                   r'(at )?(the )?([lL]eft|[lL]t\.?) temporal[ &,.]*(?!=(occipital|parietal|frontal))'
                   , ' left-temporal-lobe '
                   , text)
    #  left 'occipital'
    text = re.sub(r'([lL][Tt]\.?|[Ll]eft) O[ ,)]|(the )?(left|[lL][Tt]\.?) occipital lobe(s)?\.?|'
                   r'(at )?(the )?([lL]eft|[lL]t\.?) occipital[ &,]*(?!=(temporal|parietal|frontal))'
                   , ' left-occipital-lobe '
                   , text)
    #  left 'frontal'
    text = re.sub(r'([lL][Tt]\.?|[Ll]eft) F[ ,):]|(at )?(the )?([Ll]eft|[lL][tT]\.?) frontal( lobe|\,)|'
                   r'(at )?(the )?([lL]eft|[lL]t\.?) frontal[ &,]*(?!=(occipital|temporal|parietal))'
                   , ' left-frontal-lobe '
                   , text)
    # left 'cerebellum'
    text = re.sub(r'(the )?([lL][Tt]\.?|[lL]eft) ([cC][Bb][lL][lL]|cerebellum|cerebellar)\.?', ' left-cerebellum ', text)
    #  left 'cerebrum'
    text = re.sub(r'(the )?([lL][Tt]\.?|[lL]eft) (cerebral|cerebrum)\.?', ' left-cerebrum ', text)


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
    text = re.sub(r'([rR]ight|[lL]eft|[bB]oth)\s*orbitofrontal\s*lobe(s)?', r' \1-orbitofrontal-lobe ', text)

    return text



# Findings 데이터 전처리 작업 ( 학습에 불필요한 단어(용어)를 사전에 제거/변환하므로써 분류 성능을 높일 목적 )
# 사람이 이해하기 쉽도록 구분할 목적의 순서 기호 ( 1., 2. 등)
# 특수 문자 표현 (2개 이상의 줄넘김 또는 --> 등의 방향 표시 등)
def Findings_Preprocessing(df : pd.DataFrame, redf : pd.DataFrame) :
    cnt = 0                         # Test print count
    raw_find = []                   # Findings Raw Data List
    after_find = []                 # Findings Preprocessing List
    for i in range(df.shape[0]) :   # shape는 (Row 수, Column 수)
        if not  1811 <= i < 1831 : continue
        row = df.iloc[i]
        Ftext = ' '.join(map(str, row['Findings'].split('\n'))).strip()
        Ftext = Ftext.replace('\r', '')
        raw_data = Ftext

        ##  1. Findings에 포함된 주요 의학 용어 정형화.
        Ftext = re.sub(r'([nN]o\s)((?!at )\w+)(?:(?=\s|\.|\,))', r' \1-\2 ', Ftext)                                 # No + 용어를 1개의 토큰으로 만듦.
        Ftext = re.sub(r'([sS]tage)\s*(\d+)[,.]+', r' \1-\2 ', Ftext)                                               # stage 3,
        Ftext = re.sub(r'\(CE\)', ' contrast-enhancement ', Ftext)                                                  # (CE)
        Ftext = re.sub(r'\([nN]on CE\)', ' Non-contrast-enhancement ', Ftext)                                       # (Non CE)
        Ftext = re.sub(r'\(IA\)\-?', ' stage-ia ', Ftext)                                                           # lung(IA)-NSCLC
        Ftext = re.sub(r'[pP]\-[cC][Oo][Mm]\.?\s*a((\.|\))\)?|rtery)|'
                       r'anterior communicating artery', ' posterior-communicating-artery ', Ftext)                        # anterior communicating artery
        Ftext = re.sub(r'[pP]\-[cC][Oo][Mm](\.|\&)?|[pP][cC][oO][mM](\.|\&)?|'
                       r'posterior communicating', ' posterior-communicating ', Ftext)                                     # posterior communicating, PCOM
        Ftext = re.sub(r'[aA]\-[cC][oO][mM]\.?|[aA][cC][oO][mM]\.?|'
                       r'anterior communicating', ' anterior-communicating ', Ftext)                                       # anterior communicating, ACOM
        Ftext = re.sub(r'[Ll]arge [bB]-cell lymphoma', r'dlbcl', Ftext)                                             # large B-cell lymphoma, DLBCL
        Ftext = re.sub(r'[nN]on(-)?small cell lung cancer', r'nsclc', Ftext)                                           # Non-small cell lung cancer.
        Ftext = re.sub(r'[vV]on [Hh]ippel[- ][lL]indau [Dd]isease', r'vhl-disease', Ftext)                          # Von Hippel–Lindau disease.
        Ftext = re.sub(r'\bMM[.,]', r' multiple-myeloma ', Ftext)                                                   # MM을 길이 단위 mm으로 혼용되지 않도록.
        Ftext = re.sub(r'[iI]ntracranial [tT][oO][fF] [mM][rR][aA]', 'intracranial-tof-mra', Ftext)                 # Intracranial TOF MRA
        Ftext = re.sub(r'[Nn]eck [tT][oO][fF] [mM][rR][aA]', 'neck-tof-mra', Ftext)                                 # Neck TOF MRA
        Ftext = re.sub(r'[Nn]eck [mM][rR][aA]', 'neck-mra', Ftext)                                                  # Neck MRA
        Ftext = re.sub(r'[cC]ircle of [wW]illis', 'circle-of-willis', Ftext)
        Ftext = re.sub(r'GRE [iI]mage', 'gre-image', Ftext)
        Ftext = re.sub(r'Clinical information\s*:|\*\s*CI\s?:|CI[,:]', '', Ftext)           # Clinical information, CI:, CI,
        Ftext = re.sub(r'[sS]\/[pP]', ' status-post ', Ftext)                               # s/p, S/P
        Ftext = re.sub(r'[rR][/][oO]', ' rule-out ', Ftext)                                 # r/o, R/O
        Ftext = re.sub(r'[Ff][./-][Uu]|follow up|follow\-up|'
                              r'Fu (?=MRI)', ' follow-up ', Ftext)                                 # f/u, f-u, f.u
        Ftext = re.sub(r'N\/V', r'nausea vomiting', Ftext)                                  # N/V = Nausea and Vomiting
        Ftext = re.sub(r'[tT]2\*', r' t2-star ', Ftext)                                     # T2*
        Ftext = re.sub(r'[Tt]2\/[fF][lL][aA][iI][rR]', r' t2-flair ', Ftext)                # T2/FLAIR
        Ftext = re.sub(r'\b[wW][/-][uU]\.?\b', r'work-up', Ftext)                           # w/u, W/U.
        Ftext = re.sub(r'[gG]rade\s*(\d?|[iI]+)\s*\)?\.?', r' grade-\1 ', Ftext)            # grade 2 등
        Ftext = re.sub(r'[cC][/][Ww]', ' consistent-with ', Ftext)
        Ftext = re.sub(r'[fF]\/[iI]', ' further-investigation ', Ftext)
        Ftext = re.sub(r'\b[Nn][./-][sS]|[nN]on?( other)? [sS]ignificant|without significant change|[nN]o evidence of significant|[nN]on\s*specific'
                       ,' non-specific ', Ftext)
        Ftext = re.sub(r'with or without', r'with-or-without', Ftext)
        Ftext = re.sub(r'\s*(\-|\()?\s*[Dd][Dd][xX].?', ' ', Ftext)                                 # (DDx.
        Ftext = re.sub(r'[rR]ec\s*[\).]', ' ', Ftext)                                               # Rec)
        Ftext = re.sub(r'[dD][/][tT]|due to', ' due-to ', Ftext)                                    # d/t, due to
        Ftext = re.sub(r'\([iI][dD][xX]\s*\d+.*?\)\.?', ' ', Ftext)                                 # 영상 이미지의 인덱스와 관련된 설명은 전부 삭제.
        Ftext = re.sub(r'imaging', 'image', Ftext)                                                  # 'image' 통일.
        Ftext = re.sub(r'A(\d+)(\s|\.|에|\-|\,)(?:[sS]egment)?s?', r' A\1-segment ', Ftext)         # A1, A2 등을 segment로 구분
        Ftext = re.sub(r'P(\d+)(\s|\.|에|\-|\,)(?:[sS]egment)?s?', r' P\1-segment ', Ftext)         # P1, P2 등을 segment로 구분
        Ftext = re.sub(r'M(\d+)(\s|\.|에|\-||\,|까)(?:[sS]egment)?s?', r' M\1-segment ', Ftext)     # M1, M2 등을 segment로 구분
        Ftext = re.sub(r'V(\d+)(\s|\.|에|\-|\,)(?:[sS]egment)?s?', r' V\1-segment ', Ftext)         # V1, V2 등을 segment로 구분
        Ftext = re.sub(r'C1\,2', ' atlas-axis ', Ftext)
        Ftext = re.sub(r'ICAs', 'ICA', Ftext)
        Ftext = re.sub(r'ICHs', 'ICH', Ftext)
        Ftext = re.sub(r'PCAs', 'PCA', Ftext)
        Ftext = re.sub(r'([aA]xial|[Ss]agittal)\s*(T1WI|T2WI|FLAIR|t2-star|DWI)', r' \1-\2 ', Ftext)      # Axial T1WI, sagittal T1WI, axial T2WI 등
        Ftext = re.sub(r'op\.bed[., (]|(at )?op bed\.|op[., ]*bed(에서)', ' operative-bed ', Ftext)        # op.bed, op bed 등
        Ftext = re.sub(r'post op', r'postop', Ftext)                                                      # post op
        Ftext = re.sub(r'백질-회색질', r'white-and-gray-matter', Ftext)
        Ftext = re.sub(r'백질|[wW]hite [mM]atter', 'white-matter', Ftext)
        Ftext = re.sub(r'회색질|[gG]ray [mM]atter', 'gray-matter', Ftext)
        Ftext = re.sub(r'큰 차이|큰차이|큰 변화', r'signific-diff', Ftext)                         # 큰 차이, 큰차이 용어 통일.
        Ftext = re.sub(r'([a-zA-Z]+)\/([a-zA-Z]+)', r'\1-\2', Ftext)                           # toxic/metabolic 등 중간에 '/' 구분 문자 있는 용어 통일.
        Ftext = re.sub(r'([RL]) MCA',
                       lambda m:
                        f'right MCA' if m.group(1) == 'R' else
                        f'left MCA' if m.group(1) == 'L' else ""
                    ,   Ftext )

        ##  x. lobe 텍스트 데이터에 대한 정형화 작업
        Ftext = lobe_preprocessing(Ftext)

        ##  x. 숫자를 포함하여 부위의 번호(ex. 7th)를 표현하는 데이터 전처리
        matches = re.findall(r'(\d*)(\-*)(\d+(?:th|st|nd|rd))', Ftext)
        if matches :
            for grplist in matches:
                if grplist[0] == '' and grplist[1] == '' and grplist[2]:
                    if grplist[2][0] == '1':
                        Ftext = re.sub(r'1st', ' one-st ', Ftext)
                    elif grplist[2][0] == '2':
                        Ftext = re.sub(r'2nd', ' two-nd ', Ftext)
                    elif grplist[2][0] == '3':
                        Ftext = re.sub(r'3rd', ' three-rd ', Ftext)
                    elif grplist[2][0] == '4':
                        Ftext = re.sub(r'4th', ' four-th ', Ftext)
                    elif grplist[2][0] == '5':
                        Ftext = re.sub(r'5th', ' five-th ', Ftext)
                    elif grplist[2][0] == '6':
                        Ftext = re.sub(r'6th', ' six-th ', Ftext)
                    elif grplist[2][0] == '7':
                        Ftext = re.sub(r'7th', ' seven-th ', Ftext)
                    elif grplist[2][0] == '8':
                        Ftext = re.sub(r'8th', ' eight-th ', Ftext)
                    elif grplist[2][0] == '9':
                        Ftext = re.sub(r'9th', ' nine-th ', Ftext)
                    elif 'th' in grplist[2]:
                        Ftext = re.sub(r'\d+th', ' over-th ', Ftext)

        ## x. lesion의 개수를 전처리.
        matches = re.findall(r'(\d+) lesion', Ftext)
        if matches:
            matches = list(set(matches))
            for grplist in matches:
                if grplist[0] == '1':
                    Ftext = re.sub(fr'{grplist[0]} lesion', r' one lesion ', Ftext)
                elif grplist[0] == '2':
                    Ftext = re.sub(fr'{grplist[0]} lesion', r' two lesion ', Ftext)
                elif grplist[0] == '3':
                    Ftext = re.sub(fr'{grplist[0]} lesion', r' three lesion ', Ftext)
                elif grplist[0] == '4':
                    Ftext = re.sub(fr'{grplist[0]} lesion', r' four lesion ', Ftext)
                elif grplist[0] == '5':
                    Ftext = re.sub(fr'{grplist[0]} lesion', r' five lesion ', Ftext)
                elif grplist[0] == '6':
                    Ftext = re.sub(fr'{grplist[0]} lesion', r' six lesion ', Ftext)
                elif grplist[0] == '7':
                    Ftext = re.sub(fr'{grplist[0]} lesion', r' seven lesion ', Ftext)
                elif grplist[0] == '8':
                    Ftext = re.sub(fr'{grplist[0]} lesion', r' eight lesion ', Ftext)
                elif grplist[0] == '9':
                    Ftext = re.sub(fr'{grplist[0]} lesion', r' nine lesion ', Ftext)


        ## x. '개수'를 의미하는 수치 데이터 전처리.
        Ftext = re.sub(r'(innumerable.*?)(\d+)\)', r'\1 \2-cnt ', Ftext)        # (innumerable, > 30)


        ## x. 'ms'를 의미하는 수치 데이터 전처리.
        Ftext = re.sub(r'\b(\d+)\s*(MS|ms)\b|TE\s+(\d+)'
                       , lambda m: (
                            f"ms-{m.group(1)}" if m.group(1) else           # \1: 숫자 (ms)
                            f"te ms-{m.group(3)}" if m.group(3) else ""     # TE 144 등의 ms없는 표현
                       ), Ftext)

        ## x. % = percent 값 전처리.
        Ftext = re.sub(r'(\d+)\s*\%', r' percent-\1 ', Ftext)                   # 10%  --> percent-10

        ## Cho/NAA 수치 데이터 전처리.
        #  Cho/Cr = 2.29,  Cho/NAA = 1.14
        matches = re.findall(r'\(?(Cho\-NAA|Cho\-Cr)[ ,=]*((?:\d+(?:\.\d+)?(?:[, ]+)?)+)(?:\)\.?| at|\()', Ftext)
        Ftext = re.sub(r'\(?(Cho\-NAA|Cho\-Cr)[ ,=]*', r' ', Ftext)
        if matches:
            for grplist in matches:
                # grplist: tuple → ('Cho-NAA', '0.85 0.79 0.90')
                grp_values = grplist[1]  # 수치 문자열 전체
                values = re.findall(r"\d+(?:\.\d+)?", grp_values)
                for v in values:
                    if v == '': continue
                    tmp = re.sub(r'\.', '-', v)
                    Ftext = re.sub(fr'\b{v}\b', fr' {grplist[0]}-{tmp} ', Ftext)


        ## x. 'Cho/Cr, Cho/NAA 수치의 기준 값 전처리.
        #  Cho/Cr = 2.29 (< 2.39). Cho/NAA = 1.14 (< 1.73).
        matches = re.findall(r'(Cho\-Cr|Cho\-NAA)\-(\d+\-\d+)\s*\(?([<>])\s*(\d+(?:\.\d+))\)?\.?', Ftext)
        if matches:
            for grplist in matches:
                print(grplist)
                Ftext = re.sub(fr'(?<={grplist[0]}-{grplist[1]})\s*\(?\<\s*', r' less-than ', Ftext)
                Ftext = re.sub(fr'(?<={grplist[0]}-{grplist[1]})\s*\(?\>\s*', r' greater-than ', Ftext)
                tmp2 = grplist[3].replace('.', '-')
                Ftext = re.sub(fr'({grplist[0]}-{grplist[1]}\s*(greater|less)-than)\s*{grplist[3]}\)?\.?', fr' \1 {grplist[0]}-{tmp2} ', Ftext)


        ## x. 'ADC' 수치 데이터 전처리.
        #  ADC(avg) values of 0.862
        Ftext = re.sub(r'ADC.*value(?:s)?.*?(\d+(?:\.\d+)?)\s*\)?\.?', r' adc-\1 ', Ftext)
        Ftext = re.sub(r'(adc-\d+)\.(\d+)', r' \1-\2 ', Ftext)
        # 특수문자가 포함된 일반적인 ADC 수치 데이터 정형화.
        matches = re.finditer(r'\(?(ADC)[ ,=]*(\d+(?:\.\d+)?(?:[, ]+\d+(?:\.\d+)?)*)\)\.?', Ftext)
        #Ftext = re.sub(r'\(?ADC\s*', r' ', Ftext)                                                   # ADC 값 정형화 전 'ADC' 삭제
        if matches :
            for grplist in matches:
                grp_values = grplist.group(2)
                values = re.findall(r"\d+(?:\.\d+)?", grp_values)
                for v in values:
                    if v == '': continue
                    tmp = re.sub(r'\.', r'-', v)
                    Ftext = re.sub(fr'{v}', fr'adc-{tmp}', Ftext)


        ## 2. 양성과 음성을 구분하는 문자를 명확한 단어로 변경.
        ## (+) --> positive, (-) --> negative
        if '(+)' in Ftext :
            Ftext = re.sub(r'\(\+\)|\w\s\+', 'positive', Ftext)

        if '(-)' in Ftext:
            Ftext = re.sub(r'\(\-\)', 'negative', Ftext)

        # aneurysm의 4차원 값 표현 구분
        # Length + Width + Height(Depth) + Neck
        matches = re.findall(r'(\d{1,2}(?:\.\d{1,2})?)\s*(x|X|\*)\s*(\d{1,2}(?:\.\d{1,2})?)\s*(x|X|\*)\s*(\d(?:\.\d{1,2})?)\s*(x|X|\*)\s*(\d(?:\.\d{1,2})?)\s*\(neck\)\s*(mm|cm)', Ftext)
        if matches :
            for grplist in matches :
                if grplist[-1] == 'mm' :
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

                    Ftext = re.sub(fr'{Lvalue}\s*(x|X|\*)(?=\s*{Wvalue}\s*(x|X|\*)\s*{Hvalue}\s*(x|X|\*)\s*{Nvalue}\s*\(neck\)\s*mm)'
                                   , fr' Length-{Ltmp}mm '
                                   , Ftext)

                    Ftext = re.sub(fr'(?<=Length-{Ltmp}mm)\s*{Wvalue}\s*(x|X|\*)(?=\s*{Hvalue}\s*(x|X|\*)\s*{Nvalue}\s*\(neck\)\s*mm)'
                                   , fr' Width-{Wtmp}mm '
                                   , Ftext)

                    Ftext = re.sub(fr'(?<=Length-{Ltmp}mm Width\-{Wtmp}mm)\s*{Hvalue}\s*(x|X|\*)(?=\s*{Nvalue}\s*\(neck\)\s*mm)'
                                   , fr' Height-{Htmp}mm '
                                   , Ftext)

                    Ftext = re.sub(fr'(?<=Length-{Ltmp}mm Width-{Wtmp}mm Height-{Htmp}mm)\s*{Nvalue}\s*\(neck\)\s*mm'
                                   , fr' Neck-{Ntmp}mm '
                                   , Ftext)


        # 3차원 크기 데이터 정형화
        matches = re.findall(r'(\d{1,2}(?:\.\d{1,2})?)\s*(?:cm|mm)?(x|\*|X)(?:\.)?\s*(\d{1,2}(?:\.\d{1,2})?)\s*(?:cm|mm)?(x|\*|X)(?:\.)?\s*(\d{1,2}(?:\.\d{1,2})?)\s*(?:\(neck\))?\s*(cm|mm)',Ftext)
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
                    Ftext = re.sub(fr'{Lvalue}' + r'(?=\s*(cm)?(x|\*|X)\.?\s*(\d{1,2}(\.\d{1,2})?)\s*(cm)?(x|\*|X)\.?\s*(\d{1,2}(\.\d{1,2})?)\s*cm)'
                        ,fr' Length-{Ltmp}mm '
                        , Ftext)
                    # Width 정형화
                    Ftext = re.sub(fr'(?<=Length-{Ltmp}mm ).+{Wvalue}\s*(cm)?(?=(x|\*|X)\.?\s*{Hvalue})'
                        ,fr'Width-{Wtmp}mm '
                        , Ftext)
                    # Height 정형화
                    Ftext = re.sub(fr'(?<=Length-{Ltmp}mm Width-{Wtmp}mm ).+{Hvalue}\s*cm'
                        ,fr'Height-{Htmp}mm '
                        , Ftext)
                else:
                    Ftext = re.sub(fr'{grplist[0]}' + r'(?=\s*(mm)?(x|\*|X)\.?\s*(\d{1,2}(\.\d{1,2})?)\s*(mm)?(x|\*|X)\.?\s*(\d{1,2}(\.\d{1,2})?)\s*(\(neck\))?\s*mm)'
                        ,fr' Length-{grplist[0]}mm '
                        , Ftext)
                    Ftext = re.sub(fr'(?<=Length-{grplist[0]}mm ).+{grplist[2]}\s*(mm)?(?=(x|\*|X)\.?\s*{grplist[4]})'
                        ,fr'Width-{grplist[2]}mm '
                        , Ftext)
                    Ftext = re.sub(fr'(?<=Length-{grplist[0]}mm Width-{grplist[2]}mm ).+{grplist[4]}\s*mm'
                        ,fr'Height-{grplist[4]}mm '
                        , Ftext)
                # print(matches)
                # print(Ctext)

        # 3차원. 변경된 크기 이전의 값을 의미하는 부분의 정형화 (단위 표시가 없으며 뒤에 '-' 또는 '->', '-->', '--->' 등이 붙는다).
        # 실수로 표현된 값은 'cm' 단위이므로 'mm' 단위로 변환한다.
        matches = re.findall(r'(\d{1,2}(?:\.\d{1,2})?)\s*(x|\*|X)\s*(\d{1,2}(?:\.\d{1,2})?)\s*(x|\*|X)\s*(\d{1,2}(?:\.\d{1,2})?)\s*(\-{1,4}\>?)', Ftext)
        if matches:
            for grplist in matches:
                if '.' in grplist[0]:
                    Ltmp = str(int(float(grplist[0]) * 10))
                    Lvalue = re.sub(r'\.', r'\\.', grplist[0])
                else:
                    Ltmp, Lvalue = grplist[0], grplist[0]

                if '.' in grplist[2]:
                    Wtmp = str(int(float(grplist[2]) * 10))
                    Wvalue = re.sub(r'\.', r'\\.', grplist[2])
                else:
                    Wtmp, Wvalue = grplist[2], grplist[2]

                if '.' in grplist[4]:
                    Htmp = str(int(float(grplist[4]) * 10))
                    Hvalue = re.sub(r'\.', r'\\.', grplist[4])
                else:
                    Htmp, Hvalue = grplist[4], grplist[4]

                # Length 정형화
                Ftext = re.sub(fr'([^1-9]|^|\(){Lvalue}(?=\s*(x|\*|X)\s*{Wvalue}\s*(x|\*|X)\s*{Hvalue}' + r'\s*\-{1,4}\>?)'
                               , fr' Length-{Ltmp}mm '
                               , Ftext)

                # Width 정형화
                Ftext = re.sub(fr'(?<=Length\-{Ltmp}mm).+{Wvalue}(?=\s*(x|\*|X)\s*{Hvalue}' + r'\s*\-{1,4}\>?)'
                               , fr' Width-{Wtmp}mm '
                               , Ftext)

                # Height 정형화
                Ftext = re.sub(fr'(?<=Length\-{Ltmp}mm Width\-{Wtmp}mm).+{Hvalue}' + r'\s*\-{1,4}\>?'
                               , fr' Height-{Htmp}mm change '
                               , Ftext)
            # print(matches)
            # print(Ctext)

        # 2차원. 변경된 크기 이전의 값을 의미하는 부분의 정형화 (단위 표시가 없으며 뒤에 '-' 또는 '->', '-->', '--->' 등이 붙는다).
        # 실수로 표현된 값은 'cm' 단위이므로 'mm' 단위로 변환한다.
        matches = re.findall(r'(\d{1,2}(?:\.\d{1,2})?)\s*(x|\*|X)\s*(\d{1,2}(?:\.\d{1,2})?)(?:cm|mm)?\s*(\-{1,4}\>*)', Ftext)
        if matches:
            for grplist in matches:
                if '.' in grplist[0]:
                    Ltmp = str(int(float(grplist[0]) * 10))
                    Lvalue = re.sub(r'\.', r'\\.', grplist[0])
                else:
                    Ltmp, Lvalue = grplist[0], grplist[0]

                if '.' in grplist[2]:
                    Wtmp = str(int(float(grplist[2]) * 10))
                    Wvalue = re.sub(r'\.', r'\\.', grplist[2])
                else:
                    Wtmp, Wvalue = grplist[2], grplist[2]

                # Length 정형화
                Ftext = re.sub(fr'([^1-9]|^|\(){Lvalue}' + r'(?=\s*(x|\*|X)\s*\d{1,2}(\.\d{1,2})?(cm|mm)?\s*\-{1,4}\>*)'
                                , fr' Length-{Ltmp}mm '
                                , Ftext)

                # Width 정형화
                Ftext = re.sub(fr'(?<=Length\-{Ltmp}mm ).+{Wvalue}(cm|mm)?' + r'\s*\-{1,4}\>*'
                                , fr'Width-{Wtmp}mm change '
                                , Ftext)
                    # print(matches)
                    # print(Ctext)

        # 2차원 크기 데이터 정형화
        matches = re.findall(r'(\d{1,2}(?:\.\d{1,2})?)\s*(?:cm|mm)?(x|\*|X)(?:\.)?\s*(\d{1,2}(?:\.\d{1,2})?)\s*(?:\-)?(cm|mm|\))', Ftext)
        if matches:
            matches = list(set(matches))
            for grplist in matches:
                if grplist[-1] == 'cm' or grplist[-1] == ')':
                    Ltmp = str(int(float(grplist[0]) * 10))
                    Wtmp = str(int(float(grplist[2]) * 10))
                    Lvalue = re.sub(r'\.', r'\\.', grplist[0])
                    Wvalue = re.sub(r'\.', r'\\.', grplist[2])

                    # Length 정형화
                    Ftext = re.sub(fr'([^1-9]|^){Lvalue}' + r'(?=\s*(cm)?(x|\*|X)\.?\s*(\d{1,2}(\.\d{1,2})?)\s*\-?(cm|\)))'
                                   ,fr' Length-{Ltmp}mm '
                                   , Ftext)
                    # Width 정형화
                    Ftext = re.sub(fr'(?<=Length\-{Ltmp}mm ).+{Wvalue}\s*\-?(cm|\))'
                                   ,fr'Width-{Wtmp}mm '
                                   , Ftext)
                else:
                    # mm 단위인데 실수 형태인 경우, 반올림하여 정수화.
                    if grplist[-1] == 'mm' and '.' in grplist[0]:
                        Ltmp = str(round(float(grplist[0])))
                        Lvalue = re.sub(r'\.', r'\\.', grplist[0])
                        Ftext = re.sub(fr'([^1-9]|^){Lvalue}' + r'(?=\s*(mm)?(x|\*|X)\.?\s*(\d{1,2}(\.\d{1,2})?)\s*\-?mm)'
                                       ,fr' Length-{Ltmp}mm '
                                       , Ftext)
                    else:
                        # Length 정형화
                        Ftext = re.sub(fr'([^1-9]|^){grplist[0]}' + r'(?=\s*(mm)?(x|\*|X)\.?\s*(\d{1,2}(\.\d{1,2})?)\s*\-?mm)'
                                       ,fr' Length-{grplist[0]}mm '
                                       , Ftext)

                    if grplist[-1] == 'mm' and '.' in grplist[2]:
                        Wtmp = str(round(float(grplist[2])))
                        Wvalue = re.sub(r'\.', r'\\.', grplist[2])
                        Ftext = re.sub(fr'(?<=Length\-{Ltmp}mm ).+{Wvalue}\s*\-?mm'
                                       ,fr'Width-{Wtmp}mm '
                                       , Ftext)
                    else:
                        # Width 정형화
                        Ftext = re.sub(fr'(?<=Length\-{grplist[0]}mm ).+{grplist[2]}\s*\-?mm'
                                       ,fr'Width-{grplist[2]}mm '
                                       , Ftext)
            # print(matches)
            # print(Ctext)



        # 1차원 크기 데이터 정형화 (3차원, 2차원 크기 데이터에 대해 모두 정형화가 완료된 상태여야 한다.)
        # 이미 정형화가 완료된 텍스트는 중복 변환되지 않도록 마스킹 처리 후 마지막에 복원하는 방법으로 구현.
        global mask_matches
        mask_matches = []
        mask_pattern = re.compile(r'\b(?:Length|Width|Height)\-\d{1,2}(?:\.\d{1,2})?mm\b')
        masked = mask_pattern.sub(Mask_Repl, Ftext)  # 정형화 전 이미 정형화된 데이터는 겹치지 않도록 마스킹.

        # # 마스킹된 텍스트에서 크기 변경 전 1차원 크기 데이터 추출 (ex. 1.5 - 2.1cm 형태에서 1.5 값)
        matches = re.findall(r'(\d{1,2}(?:\.\d{1,2})?)(?:cm|mm)?\s*(\-{1,5}>?)[ 0-9.]*?(cm|mm)', masked)
        if matches:
            # 중복되는 Group이 2번 이상 정형화되지 않도록 세트화.
            matches = list(set(matches))
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
                        masked = re.sub(fr'{Lvalue}(cm)?\s*\-+\>', fr' Length-{Ltmp}mm change ', masked)
                    elif grplist[-1] == 'mm':
                        if '.' in grplist[0]:
                            Ltmp = str(int(round(float(grplist[0]))))
                            Lvalue = re.sub(r'\.', r'\\.', grplist[0])
                            masked = re.sub(fr'{Lvalue}\s*\s*\-+\>', fr' Length-{Ltmp}mm change ', masked)
                        else:
                            masked = re.sub(fr'{grplist[0]}\s*\s*\-+\>', fr' Length-{grplist[0]}mm change ', masked)

                # 마스킹된 텍스트를 복원 후, 최종 결과를 Ctext에 저장.
                for token, text in mask_matches:
                    masked = masked.replace(token, text)

                Ftext = masked
                masked = mask_pattern.sub(Mask_Repl, Ftext)

        mask_matches = []
        mask_pattern = re.compile(r'\b(?:Length|Width|Height)\-\d{1,2}(?:\.\d{1,2})?mm\b')
        masked = mask_pattern.sub(Mask_Repl, Ftext)

        # 마스킹된 텍스트에서 1차원 크기 데이터 추출.
        matches = re.findall(r'(\d{1,2}(?:[.,]\d{1,2})?)\s*(mm|cm)[가-힣]*\b', masked)
        if matches:
            # 중복되는 Group에 의해 2번 정형화되지 않도록 세트화.
            matches = list(set(matches))
            for grplist in matches:
                if grplist[-1] == 'cm':
                    Ltmp = str(int(float(grplist[0]) * 10))
                    Lvalue = re.sub(r'\.', r'\\.', grplist[0])
                    masked = re.sub(fr'([^1-9]|^|\(){Lvalue}\s*cm', fr' Length-{Ltmp}mm ', masked)
                elif grplist[-1] == 'mm' and '.' in grplist[0]:
                    # Ltmp = re.sub(r'(\d{1,2}).+', r'\1', grplist[0])
                    Lvalue = re.sub(r'\.', r'\\.', grplist[0])
                    Ltmp = str(round(float(grplist[0])))
                    masked = re.sub(fr'([^1-9]|^|\(|(?<!Length\-))({Lvalue}\s*mm)', fr' Length-{Ltmp}mm ', masked)
                elif grplist[-1] == 'mm' and ',' in grplist[0]:
                    Ltmp = str(round(float(grplist[0].replace(',', '.'))))
                    masked = re.sub(fr'([^1-9]|^|\(|(?<!Length\-))({grplist[0]}\s*mm)', fr' Length-{Ltmp}mm ', masked)
                else:
                    masked = re.sub(fr'([^1-9]|^|\(|(?<!Length\-)){grplist[0]}\s*mm[가-힣]*', fr' Length-{grplist[0]}mm', masked)

                # 마스킹된 텍스트를 복원 후, 최종 결과를 Ctext에 저장.
                for token, text in mask_matches:
                    masked = masked.replace(token, text)

                Ftext = masked
                masked = mask_pattern.sub(Mask_Repl, Ftext)

        #  두 크기(길이) 값 사이에 존재하는 '변동'을 의미하는 특수 문자의 정형화 처리.
        Ftext = re.sub(r'(?<=mm)\s*\-+\>\s*(?=Length\-)', ' change ', Ftext)


        ##  4. 날짜 기록 데이터 (2011.07.08.), (2011. 11. 11.), (2004) 전체 삭제.
        #   날짜 데이터는 '이전'의 의미를 전달할 뿐, 크게 의미 있지 않다고 판단하여 텍스트 삭제 진행.
        Ftext = re.sub(r'\(?\d{4}\.\d{1,2}\.\d{1,2}\.?\)?|'
                       r'\(\d{4}\)|'
                       r'\(?\d{4}[.\- ]*\d{1,2}[.\- ]*\d{1,2}[.\-)]*|'
                       r'on (\d+\/\d+\/\d{4}|\d{1,2}\/\d{1,2})\)?\.?|'                  # on 5/8/2024, on 5/16  
                       r'in \d{4}(\)\.)?|'                                              # in 2024).
                       r'\d+년\s*\d{1,2}월\s*\d{1,2}일'                                  # 2020년 6월 10일
                       , ' ', Ftext)

        ##  x.특정 구분 텍스트 삭제 (DDX, Note 등)
        Ftext = re.sub(r'\-\s*[iI][dD][xX][0-9 ,]*[iI][mM][0-9 ,]*[iI][dD][xX][0-9 ,]*[iI][mM][0-9 ,]*[.,\- ]?|'
                       r'[\[ ]*I[Dd][Xx][ 0-9]*I[Mm][ 0-9]*[\] ,]*'
                       , ' ', Ftext)                                               # 영상 이미지 인덱스 번호 표현  ([IDX 4 IM 17].)
        Ftext = re.sub(r'\d+ (yr|years) ago', ' ', Ftext)                   # 2 yr ago
        Ftext = re.sub(r'\bI+\.\s', ' ', Ftext)                             # I., II., III.,
        Ftext = re.sub(r'\*?\s*[nN]ote\s*[,:.]', ' ', Ftext)                # * Note:
        Ftext = re.sub(r'(?<=[a-zA-Z가-힣\)])\.(\s|$|\n|\t)', ' ', Ftext)    # 문장의 마지막 '.'
        Ftext = re.sub(r'\d\.\s', ' ', Ftext)                               # 1., 2., 3.,
        Ftext = re.sub(r'[(\[]\d[)\]]\:?', ' ', Ftext)                      # (1), (2), [1], [2] ...
        Ftext = re.sub(r'(?<=\w)[`\'\’]([Ss]|\s)*', ' ', Ftext)             # Parkinson's
        Ftext = re.sub(r'\s(\-+|\(|\))\s', ' ', Ftext)                      # 구분 문자 역할의 ' - ' 등.
        Ftext = re.sub(r'\s\d\)[.,]?\s', ' ', Ftext)                        # 순서 번호 역할의 1), 2).




        ##  7. 대소 비교 문자('<', '>') 텍스트 변환.
        #   '<', '>'는 다른 특수 문자와 조합하여 '구분 문자' 역할로 사용하는 경우도 있다.
        #   이를 제외(삭제)하면 크기나 백분율 비교로 사용하므로 이들에 대한 명확한 텍스트 변환이 필요하다.
        #   대소 구분 목적으로 사용하는 기호 표현을 'less than', 'greater than'으로 텍스트 변경한다.
        Ftext = re.sub(r'\(?[lL]t\s', ' left ', Ftext)
        Ftext = re.sub(r'[rR]t[ )]', ' right ', Ftext)
        Ftext = re.sub(r'Lt\.?\s*\>\s*Rt\.?|Rt\.?\s*\<\s*Lt\.?|[lL]eft\s*\>\s*[rR]ight|[rR]ight\s*\<\s*[lL]eft'
                       , ' left greater-than right '
                       , Ftext)
        Ftext = re.sub(r'Rt\.?\s*\>\s*Lt\.?|Lt\.?\s*\<\s*Rt\.?|[rR]ight\s*\>\s*[lL]eft|[lL]eft\s*\<\s*[rR]ight'
                       , ' right greater-than left '
                       , Ftext)
        Ftext = re.sub(r'(?<!temporal)(?<!MRA)(?<!\-)>\s*(?=[rR]ight|[Ll]eft|[rR]t|[lL]t|\d|[gG]rade|Length)|greater than'
                       , ' greater-than '
                       , Ftext)                                                                                    #
        Ftext = re.sub(r'\(?<\s*(?=[rR]ight|[Ll]eft|[rR]t|[lL]t|\d|[gG]rade|Length)|less than'
                       , ' less-than '
                       , Ftext)                                                                                     # (< Length-5mm, (< Grade, ...
        Ftext = re.sub(r'<(?=[a-zA-Z가-힣])|(?<=[a-zA-Z가-힣])>', ' ', Ftext)                              # <Brain, dings>
        Ftext = re.sub(r'(?<=[a-zA-Z가-힣])\s*[,:;]', ' ', Ftext)                                         # DWI,
        # Ftext = re.sub(r'<', 'less than', Ftext)

        ## x. 구분자 역할의 특수문자 제거
        Ftext = re.sub(r'\((?=[a-zA-Z])|(?<=[a-zA-Z])\)|'
                       r'[-=]+>|'
                       r'\*\.\s*\*\s*', ' ', Ftext)


        token_test = tokenizer_bert.tokenize(Ftext)  # 전처리한 소견을 토큰화
        token_test = merge_wordpieces(token_test)  # 토큰 데이터를 재결합
        token_test = reorg_wordpieces(token_test)  # 토큰 데이터의 불용어 제거 및 영문+한글 단어의 정형화

        ## 8. 2회 이상의 띄어쓰기 또는 줄바꿈 문자에 대해 한 번의 줄바꿈만 적용.
        #print(f"Start conv\n{Ftext}")
        Ftext = re.sub(r'\s{2,20}', ' ', Ftext)
        print('##############################################')
        print(raw_data)
        print('-------' * 20)
        # print(Ftext)
        # print('-------' * 20)
        print(token_test)
        print('##############################################')

        # matches = re.findall(r'\.|\,', Ctext)
        # cnt += 1
        # if cnt == 30 : break

        ## 모든 Raw Data 탐색 결과를 저장 후 return.
        raw_find.append(raw_data)
        after_find.append(Ftext)

        redf.loc[i, 'finding'] = Ftext

        # 테스트 목적의 반복문 도중 반환.
        # if cnt == 100 :
        #     break

    return raw_find, after_find


# Conclusion 데이터 전처리 작업 ( 학습에 불필요한 단어(용어)를 사전에 제거/변환하므로써 분류 성능을 높일 목적 )
def Conclusion_Preprocessing(df : pd.DataFrame, redf : pd.DataFrame) :
    cnt = 0         # Test print count
    raw_conc = []   # Conclusion Raw Data List
    after_conc = [] # Conclusion Preprocesing List

    for i in range(df.shape[0]) :   # shape() = (Row 수, Column 수)
        row = df.iloc[i]
        Ctext = ' '.join(map(str, row['Conclusion'].split('\n'))).strip()
        Ctext = Ctext.replace('\r', '')
        raw_data = Ctext

        ## x. 특수문자가 포함된 의학 용어 변환.
        Ctext = re.sub(r'[pP]\-[cC][Oo][Mm]\.?\s*a((\.|\))\)?|rtery)', ' posterior-communicating-artery ', Ctext)
        Ctext = re.sub(r'[pP]\-[cC][Oo][Mm](\.|\&)?|[pP][cC][oO][mM](\.|\&)?', ' posterior-communicating ', Ctext)
        Ctext = re.sub(r'\(CE\)', ' contrast-enhancement ', Ctext)
        Ctext = re.sub(r'\([nN]on CE\)', ' Non-contrast-enhancement ', Ctext)
        Ctext = re.sub(r'op\.bed[., (]|(at )?op bed\.', ' operative-bed ', Ctext)
        Ctext = re.sub(r'[Ff][./-][Uu]|follow up|follow\-up', ' follow-up ', Ctext)
        Ctext = re.sub(r'[Nn][./-][sS]|[nN]on?( other)? [sS]ignificant|without significant change|[nN]o evidence of significant|[nN]onspecific', ' non-specific ', Ctext)
        Ctext = re.sub(r'op\.site[., (]|(at )?(the )?op site', ' operative-site ', Ctext)
        Ctext = re.sub(r'postop\.change[., (]', ' postoperative-change ', Ctext)
        Ctext = re.sub(r'e\.g\.?', ' ', Ctext)
        Ctext = re.sub(r'[Jj][Xx]\.', ' junction ', Ctext)
        Ctext = re.sub(r'[sS]\/[pP]', ' status-post ', Ctext)
        Ctext = re.sub(r'[rR][/][oO]', ' rule-out ', Ctext)
        Ctext = re.sub(r'[dD][/][tT]|due to', ' due-to ', Ctext)
        Ctext = re.sub(r'\b[wW][/-][uU]\.?\b', r'work-up', Ctext)
        Ctext = re.sub(r'\/[cC]', ' with ', Ctext)
        Ctext = re.sub(r'[aA]\-[cC][oO][mM]\.?|[aA][cC][oO][mM]\.?', ' anterior communicating ', Ctext)
        Ctext = re.sub(r'[cC][/][Ww]', ' consistent-with ', Ctext)
        Ctext = re.sub(r'[fF]\/[iI]', ' further-investigation ', Ctext)
        Ctext = re.sub(r'Lt\.?\s*\>\s*Rt\.?|Rt\.?\s*\<\s*Lt\.?|[lL]eft\s*\>\s*[rR]ight|[rR]ight\s*\<\s*[lL]eft', ' left greater-than right', Ctext)
        Ctext = re.sub(r'Rt\.?\s*\>\s*Lt\.?|Lt\.?\s*\<\s*Rt\.?|[rR]ight\s*\>\s*[lL]eft|[lL]eft\s*\<\s*[rR]ight', ' right greater-than left', Ctext)
        Ctext = re.sub(r'\(\+\)', ' positive ', Ctext)
        Ctext = re.sub(r'\(\-\)', ' negative ', Ctext)
        Ctext = re.sub(r'C1\,2', ' atlas-axis ', Ctext)
        Ctext = re.sub(r'T(\d)\*', r'T\1-star', Ctext)
        Ctext = re.sub(r'ICAs', 'ICA', Ctext)
        Ctext = re.sub(r'ICHs', 'ICH', Ctext)
        Ctext = re.sub(r'PCAs', 'PCA', Ctext)
        Ctext = re.sub(r'A(\d+)(\s|\.|에|\-)(?:[sS]egment)?s?', r' A\1-segment ', Ctext)  # A1, A2 등을 segment로 구분
        Ctext = re.sub(r'P(\d+)(\s|\.|에|\-)(?:[sS]egment)?s?', r' P\1-segment ', Ctext)  # P1, P2 등을 segment로 구분
        Ctext = re.sub(r'M(\d+)(\s|\.|에|\-)(?:[sS]egment)?s?', r' M\1-segment ', Ctext)  # M1, M2 등을 segment로 구분
        Ctext = re.sub(r'V(\d+)(\s|\.|에|\-)(?:[sS]egment)?s?', r' V\1-segment ', Ctext)  # V1, V2 등을 segment로 구분

        ##  x. lobe 텍스트 데이터에 대한 정형화 작업
        Ctext = lobe_preprocessing(Ctext)





        ##  x. 소견에 포함된 모든 수치(크기) 데이터를 하나의 공통 형식으로 정형화 작업.
        #   규칙1 : mm 단위로 통일. 수치와 단위를 띄어쓰지 않고 붙여서 표현. 수치와 단위 사이에 표함된 모든 특수 문자는 변환 또는 제거.
        #   규칙2 : 1차원 이상으로 표현된 수치(크기) 데이터는 'x', '*' 등의 특수문자로 표현되었는데, 이를 'Length', 'Width', 'Height'로 변환.
        #           ex) 1.2mm = Length 1.2mm
        #           ex) 1.2x2.5mm = Length 1.2mm Width 2.5mm
        #           ex) 1.2x2.5x1.8mm = Length 1.2mm Width 2.5mm Height 1.8mm

        # 3차원 크기 데이터 정형화
        matches = re.findall(r'(\d{1,2}(?:\.\d{1,2})?)\s*(?:cm|mm)?(x|\*|X)(?:\.)?\s*(\d{1,2}(?:\.\d{1,2})?)\s*(?:cm|mm)?(x|\*|X)(?:\.)?\s*(\d{1,2}(?:\.\d{1,2})?)\s*(cm|mm)', Ctext)
        if matches :
            for grplist in matches :                            # 매칭된 그룹 리스트 순환. 6개의 원소가 하나의 그릅에 포함.
                if grplist[-1] == 'cm':                         # cm 단위라면 mm 단위로 변환 (cm 단위는 소수점이 포함되지만, mm 단위는 정수만으로 표현 가능).
                    Ltmp = str(int(float(grplist[0]) * 10))     # 정형화 값으로 사용할 mm단위의 L,W,H 값.
                    Wtmp = str(int(float(grplist[2]) * 10))
                    Htmp = str(int(float(grplist[4]) * 10))
                    Lvalue = re.sub(r'\.', r'\\.', grplist[0])     # Length Value : 특수문자 '.'를 정규표현식에서 일반 문자로 보이도록 변경.
                    Wvalue = re.sub(r'\.', r'\\.', grplist[2])     # Width Value.
                    Hvalue = re.sub(r'\.', r'\\.', grplist[4])     # Height Value.

                    # Length 정형화
                    Ctext = re.sub(fr'{Lvalue}' + r'(?=\s*(cm)?(x|\*|X)\.?\s*(\d{1,2}(\.\d{1,2})?)\s*(cm)?(x|\*|X)\.?\s*(\d{1,2}(\.\d{1,2})?)\s*cm)',
                                   fr' Length-{Ltmp}mm ',
                                   Ctext)
                    # Width 정형화
                    Ctext = re.sub(fr'(?<=Length-{Ltmp}mm ).+{Wvalue}\s*(cm)?(?=(x|\*|X)\.?\s*{Hvalue})',
                                   fr'Width-{Wtmp}mm '
                                   , Ctext)
                    # Height 정형화
                    Ctext = re.sub(fr'(?<=Length-{Ltmp}mm Width-{Wtmp}mm ).+{Hvalue}\s*cm',
                                   fr'Height-{Htmp}mm '
                                   , Ctext)
                else :
                    Ctext = re.sub(fr'{grplist[0]}'+r'(?=\s*(mm)?(x|\*|X)\.?\s*(\d{1,2}(\.\d{1,2})?)\s*(mm)?(x|\*|X)\.?\s*(\d{1,2}(\.\d{1,2})?)\s*mm)',
                                   fr' Length-{grplist[0]}mm ',
                                   Ctext)
                    Ctext = re.sub(fr'(?<=Length-{grplist[0]}mm ).+{grplist[2]}\s*(mm)?(?=(x|\*|X)\.?\s*{grplist[4]})',
                                   fr'Width-{grplist[2]}mm ',
                                   Ctext)
                    Ctext = re.sub(fr'(?<=Length-{grplist[0]}mm Width-{grplist[2]}mm ).+{grplist[4]}\s*mm',
                                   fr'Height-{grplist[4]}mm ',
                                   Ctext)
                # print(matches)
                # print(Ctext)


        # 3차원. 변경된 크기 이전의 값을 의미하는 부분의 정형화 (단위 표시가 없으며 뒤에 '-' 또는 '->', '-->', '--->' 등이 붙는다).
        # 실수로 표현된 값은 'cm' 단위이므로 'mm' 단위로 변환한다.
        matches = re.findall(r'(\d{1,2}(?:\.\d{1,2})?)\s*(x|\*|X)\s*(\d{1,2}(?:\.\d{1,2})?)\s*(x|\*|X)\s*(\d{1,2}(?:\.\d{1,2})?)\s*(\-{1,4}\>?)', Ctext)
        if matches :
            for grplist in matches :
                if '.' in grplist[0] :
                    Ltmp = str(int(float(grplist[0]) * 10))
                    Lvalue = re.sub(r'\.', r'\\.', grplist[0])
                else :
                    Ltmp, Lvalue = grplist[0], grplist[0]

                if '.' in grplist[2] :
                    Wtmp = str(int(float(grplist[2]) * 10))
                    Wvalue = re.sub(r'\.', r'\\.', grplist[2])
                else:
                    Wtmp, Wvalue = grplist[2], grplist[2]

                if '.' in grplist[4] :
                    Htmp = str(int(float(grplist[4]) * 10))
                    Hvalue = re.sub(r'\.', r'\\.', grplist[4])
                else :
                    Htmp, Hvalue = grplist[4], grplist[4]

                # Length 정형화
                Ctext = re.sub(fr'([^1-9]|^|\(){Lvalue}(?=\s*(x|\*|X)\s*{Wvalue}\s*(x|\*|X)\s*{Hvalue}' + r'\s*\-{1,4}\>?)',
                                fr' Length-{Ltmp}mm ',
                                Ctext)

                # Width 정형화
                Ctext = re.sub(fr'(?<=Length {Ltmp}mm).+{Wvalue}(?=\s*(x|\*|X)\s*{Hvalue}' + r'\s*\-{1,4}\>?)',
                               fr' Width-{Wtmp}mm ',
                               Ctext)

                # Height 정형화
                Ctext = re.sub(fr'(?<=Length {Ltmp}mm Width {Wtmp}mm).+{Hvalue}' + r'\s*\-{1,4}\>?',
                               fr' Height-{Htmp}mm change ',
                               Ctext)
            # print(matches)
            # print(Ctext)

        # 2차원. 변경된 크기 이전의 값을 의미하는 부분의 정형화 (단위 표시가 없으며 뒤에 '-' 또는 '->', '-->', '--->' 등이 붙는다).
        # 실수로 표현된 값은 'cm' 단위이므로 'mm' 단위로 변환한다.
        matches = re.findall(
                r'(\d{1,2}(?:\.\d{1,2})?)\s*(x|\*|X)\s*(\d{1,2}(?:\.\d{1,2})?)(?:cm|mm)?\s*(\-{1,4}\>*)', Ctext)
        if matches:
            for grplist in matches:
                if '.' in grplist[0]:
                    Ltmp = str(int(float(grplist[0]) * 10))
                    Lvalue = re.sub(r'\.', r'\\.', grplist[0])
                else:
                    Ltmp, Lvalue = grplist[0], grplist[0]

                if '.' in grplist[2]:
                    Wtmp = str(int(float(grplist[2]) * 10))
                    Wvalue = re.sub(r'\.', r'\\.', grplist[2])
                else:
                    Wtmp, Wvalue = grplist[2], grplist[2]

                # Length 정형화
                Ctext = re.sub(
                        fr'([^1-9]|^|\(){Lvalue}' + r'(?=\s*(x|\*|X)\s*\d{1,2}(\.\d{1,2})?(cm|mm)?\s*\-{1,4}\>*)',
                        fr' Length-{Ltmp}mm ',
                        Ctext)

                # Width 정형화
                Ctext = re.sub(fr'(?<=Length {Ltmp}mm ).+{Wvalue}(cm|mm)?' + r'\s*\-{1,4}\>*',
                            fr'Width-{Wtmp}mm change ',
                            Ctext)
            # print(matches)
            # print(Ctext)


        # 2차원 크기 데이터 정형화
        matches = re.findall(r'(\d{1,2}(?:\.\d{1,2})?)\s*(?:cm|mm)?(x|\*|X)(?:\.)?\s*(\d{1,2}(?:\.\d{1,2})?)\s*(?:\-)?(cm|mm|\))', Ctext)
        if matches :
            for grplist in matches :
                if grplist[-1] == 'cm' or grplist[-1] == ')':
                    Ltmp = str(int(float(grplist[0]) * 10))
                    Wtmp = str(int(float(grplist[2]) * 10))
                    Lvalue = re.sub(r'\.', r'\\.', grplist[0])
                    Wvalue = re.sub(r'\.', r'\\.', grplist[2])
                    # Length 정형화
                    Ctext = re.sub( fr'([^1-9]|^){Lvalue}' + r'(?=\s*(cm)?(x|\*|X)\.?\s*(\d{1,2}(\.\d{1,2})?)\s*\-?(cm|\)))',
                                    fr' Length-{Ltmp}mm '
                                    , Ctext)
                    # Width 정형화
                    Ctext = re.sub( fr'(?<=Length {Ltmp}mm ).+{Wvalue}\s*\-?(cm|\))',
                                    fr'Width-{Wtmp}mm '
                                    , Ctext)
                else :
                    # mm 단위인데 실수 형태인 경우, 반올림하여 정수화.
                    if grplist[-1] == 'mm' and '.' in grplist[0]:
                        Ltmp = str(round(float(grplist[0])))
                        Lvalue = re.sub(r'\.', r'\\.', grplist[0])
                        Ctext = re.sub(fr'([^1-9]|^){Lvalue}' + r'(?=\s*(mm)?(x|\*|X)\.?\s*(\d{1,2}(\.\d{1,2})?)\s*\-?mm)',
                                       fr' Length-{Ltmp}mm ',
                                       Ctext)
                    else :
                        # Length 정형화
                        Ctext = re.sub(
                            fr'([^1-9]|^){grplist[0]}' + r'(?=\s*(mm)?(x|\*|X)\.?\s*(\d{1,2}(\.\d{1,2})?)\s*\-?mm)',
                            fr' Length-{grplist[0]}mm '
                            , Ctext)

                    if grplist[-1] == 'mm' and '.' in grplist[2]:
                        Wtmp = str(round(float(grplist[2])))
                        Wvalue = re.sub(r'\.', r'\\.', grplist[2])
                        Ctext = re.sub(fr'(?<=Length {Ltmp}mm ).+{Wvalue}\s*\-?mm',
                                       fr'Width-{Wtmp}mm '
                                       , Ctext)
                    else :
                        # Width 정형화
                        Ctext = re.sub(fr'(?<=Length {grplist[0]}mm ).+{grplist[2]}\s*\-?mm',
                                       fr'Width-{grplist[2]}mm '
                                       , Ctext)
            # print(matches)
            # print(Ctext)



        # 1차원 크기 데이터 정형화 (3차원, 2차원 크기 데이터에 대해 모두 정형화가 완료된 상태여야 한다.)
        # 이미 정형화가 완료된 텍스트는 중복 변환되지 않도록 마스킹 처리 후 마지막에 복원하는 방법으로 구현.
        global mask_matches
        mask_matches = []
        mask_pattern = re.compile(r'\b(?:Length|Width|Height)\-\d{1,2}(?:\.\d{1,2})?mm\b')
        masked = mask_pattern.sub(Mask_Repl, Ctext)  # 정형화 전 이미 정형화된 데이터는 겹치지 않도록 마스킹.

        # # 마스킹된 텍스트에서 크기 변경 전 1차원 크기 데이터 추출 (ex. 1.5 - 2.1cm 형태에서 1.5 값)
        matches = re.findall(r'(\d{1,2}(?:\.\d{1,2})?)(?:cm|mm)?\s*(\-{1,5}>?)[ 0-9.]*?(cm|mm)', masked)
        if matches :
            # 중복되는 Group이 2번 이상 정형화되지 않도록 세트화.
            matches = list(set(matches))
            for grplist in matches :
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
                        else :
                            masked = re.sub(fr'{grplist[0]}\-mm[.,]?', fr' Length-{grplist[0]}mm ', masked)
                # 크기 변동 이전의 값 처리
                elif '>' in grplist[1]:
                    if grplist[-1] == 'cm':
                        Ltmp = str(int(float(grplist[0]) * 10))
                        Lvalue = re.sub(r'\.', r'\\.', grplist[0])
                        masked = re.sub(fr'{Lvalue}(cm)?\s*\-+\>', fr' Length-{Ltmp}mm change ', masked)
                    elif grplist[-1] == 'mm':
                        if '.' in grplist[0]:
                            Ltmp = str(int(round(float(grplist[0]))))
                            Lvalue = re.sub(r'\.', r'\\.', grplist[0])
                            masked = re.sub(fr'{Lvalue}\s*\s*\-+\>', fr' Length-{Ltmp}mm change ', masked)
                        else :
                            masked = re.sub(fr'{grplist[0]}\s*\s*\-+\>', fr' Length-{grplist[0]}mm change ', masked)

                # 마스킹된 텍스트를 복원 후, 최종 결과를 Ctext에 저장.
                for token, text in mask_matches:
                    masked = masked.replace(token, text)

                Ctext = masked
                masked = mask_pattern.sub(Mask_Repl, Ctext)
                #print(grplist)

        mask_matches = []
        mask_pattern = re.compile(r'\b(?:Length|Width|Height)\-\d{1,2}(?:\.\d{1,2})?mm\b')
        masked = mask_pattern.sub(Mask_Repl, Ctext)

        # 마스킹된 텍스트에서 1차원 크기 데이터 추출.
        matches = re.findall(r'(\d{1,2}(?:\.\d{1,2})?)\s*(mm|cm)\b', masked)
        if matches :
            # 중복되는 Group에 의해 2번 정형화되지 않도록 세트화.
            matches = list(set(matches))
            for grplist in matches :
                if grplist[-1] == 'cm' :
                    Ltmp = str(int(float(grplist[0]) * 10))
                    Lvalue = re.sub(r'\.', r'\\.', grplist[0])
                    masked = re.sub(fr'([^1-9]|^|\(){Lvalue}\s*cm', fr' Length-{Ltmp}mm ', masked)
                elif grplist[-1] == 'mm' and  '.' in grplist[0] :
                    #Ltmp = re.sub(r'(\d{1,2}).+', r'\1', grplist[0])
                    Ltmp = str(round(float(grplist[0])))
                    Lvalue = re.sub(r'\.', r'\\.', grplist[0])
                    masked = re.sub(fr'([^1-9]|^|\(|(?<!Length\-))({Lvalue}\s*mm)', fr' Length-{Ltmp}mm ', masked)
                else :
                    masked = re.sub(fr'([^1-9]|^|\(|(?<!Length\-)){grplist[0]}\s*mm', fr' Length-{grplist[0]}mm', masked)

                # 마스킹된 텍스트를 복원 후, 최종 결과를 Ctext에 저장.
                for token, text in mask_matches:
                    masked = masked.replace(token, text)

                Ctext = masked
                masked = mask_pattern.sub(Mask_Repl, Ctext)

        #print(Ctext)

        # 실수 표현된 텍스트 데이터 찾아보기
        # matches = re.findall(r'\d{1,2}\.\d{1,2}', Ctext)
        # if matches and len(matches) >= 2:
        #     print(matches)
        #     print(Ctext)


        ##  2. 날짜 기록 데이터 삭제
        #   '이전'의 의미를 포함할 뿐 날짜 데이터 자체가 판독 결과에 큰 영향을 주지 않으므로 삭제.
        Ctext = re.sub(r'on \d{4}(\.|\-)\d{1,2}(\.|\-)\d{1,2}\,?|'                   # on 2021.10.3, 2010-11-12 
                       r'on \d{1,2}\/\d{1,2}\)*|'                                           # on 6/16 
                       r'in\s\d{4}(\.\d{1,2}\.\d{1,2})?|'                                   # in 2010, in 2020.01.10
                       r'\(\d{4}\.\s*\d{1,2}\.\s*\d{1,2}\,\s*\d{1,2}\.\s*\d{1,2}\.\,|'      # (2021. 12. 7, 6. 20.,
                       r'\(\d{4}\.\d{1,2}\.\d{1,2}\s*\/\s*\d{4}\.\d{1,2}\.\d{1,2}\)\.*|'    # (2021.10.4 / 2022.8.10)
                       r'\(20\d{2}(?=\w)|'                                                  # (2011s
                        r'\(\d{4}\.\d{1,2}\)?\.(\d{1,2}\)(\,|\.))|'
                        r'\(\d{4}\.\d{1,2}\.\s|'
                        r'\(\d{2,4}\.\s?\d{1,2}\.\s?\d{1,2}\.?(\)|\,)(\.|\,|\\)?|'          # (2012. 11. 9).
                        r'\(\d{4}\.\,|'
                        r'\(?\d{4}(\-|\.)\d{1,2}(\-|\.)\d{1,2}\)?|'
                        r'\(\d{4}\-\d{1,2}\-\d{1,2}\s*(\-\-)?|'
                        r'\((20|1)\d{2,5}\.\)\.?\\?|'
                        r'\(\d{4}\.\d{1,2}\.\d{1,2}\s*(?=[a-zA-Z가-힣])|'
                        r'\(\d{4,5}\,\s\d{1,2}\.\,|'
                        r'\(\d{4}\.\d{1,2}\.\d{1,2}\s*\/\s*\d{4}\.\d{1,2}\.\d{1,2}\)(\.|\,)|'
                        r'\d{2,4}\s*\/\s*\d{4}\.\d{1,2}\.\d{1,2}\)(\.|\,)|'
                        r'\(\d{4}\.\d{1,2}\.\d{1,2}\s*\-\-?\>?|'
                        r'\(\d{4}\.\d{1,2}\)($|\.)', ' ', Ctext)

        ##  x. 괄호 안의 텍스트 중 크기 관련(숫자 + mm/cm), 영상 인덱스 관련(IDX, Img), 날짜 관련(2021.01.19) 데이터는 전부 삭제
        #   괄호 안에 내용은 작성된 소견에서 '보충'하는 의미로 수치나 참고 번호를 나타낸다.
        #   하지만, 그런 데이터가 판독 분류를 어렵게 만들 수 있으므로 괄호에 포함된 수치/날짜/영상번호 등의 내용은 전부 삭제한다.
        #   그 외에는 추가 정보로 사용할 수 있으므로 별도로 전처리 작업을 진행한다.
        Ctext = re.sub(r'(\(|\[).*?(I[dD][xX]|Img|I[mM]).*(\)|\])|'
                       r'\-\s*[iI][dD][xX][0-9 ,]*[iI][mM][0-9 ,]*[iI][dD][xX][0-9 ,]*[iI][mM][0-9 ,]*[.,\- ]?|'
                       r'\(20\d{2}(\.|\-).*\)\.?|'
                       r'[dD][dD]x[.:]\)?|'
                       r'rec\)|'
                       r'etc\.\)', ' ', Ctext)

        ##  1. Conclusion에 포함된 순서 번호 기호 삭제
        #   순서 번호는 가독성 개선을 목적이지 판독 결과에 영항을 주지 않는다.
        Ctext = re.sub(r'\d(\.|\))\)?(?=\s|[a-zA-Z])', ' ', Ctext)


        ##  3. 불필요한 특수 문자 조합 제거.
        Ctext = re.sub(r'(\)|\.)?\;|\:|\"|\s{2,3}\-\-?\>?|\[|\]|\.?\&|'
                       r'(?<=[a-zA-Z가-힣1-9 ])\s*(\(|\))\s*(\.|\,|\)){0,2}(?=[a-zA-Z가-힣1-9<> ]|$)|'
                       r'\.?(\-|\=){1,4}>|\s(\-|\=){2,4}(\s|\w)|'   # .-->, ==>
                       r'\(\=|\&|\(\.|\.\)|'
                       r'(?<=\w)\.(\s|$)|'
                       r'(?<=[A-Z])\.(?=[A-Z])|'
                       r'\d{1,2}\.\d{1,2}\-m{1}|'                   # 1.3-m 형태는 자체 삭제함.
                       r'\s(\*|\=)\s|'
                       r'\s+-+(\s+|$)|'
                       r'\s{1,5}\-+(?=\w)|'
                       r'\*{2,3}(\d{1,2}[.,]\d{1,2})*|^\*|'
                       r'\(\*+\)|'                                          # (**)
                       r'\<\-|'                                             # <-
                       r'(?<=[\w가-힣])[,.](?=[a-zA-Z _&가-힣]|$)|'     # I.M,  동,사
                       r'\x07|[`\']s|'                                      # 's
                       r'\(\-|\)\:|'                                        # (-, ):
                       r'\.\s{1,4}(\-|\+)'
                       , ' ', Ctext)

        # 대부분의 전처리를 완료 후 마무리 전처리 단계
        Ctext = re.sub(r'\>\s*(?=\d{1,2}.*?(mm|\%)|Length)|greater than', ' greater-than', Ctext)
        Ctext = re.sub(r'\<\s*(?=\d{1,2}.*?(mm|\%)|Length)|less than', ' less-than', Ctext)

        Ctext = re.sub(r' - |[ ,.*,][,.]|\(\*+\)\,?|^<|^>|(?<=\w)(\>|\<)\s*|→|\?+\)|^\#|\+|(?<=\w)\s*(\>|\<)\s*(?=\w)', ' ', Ctext)
        Ctext = re.sub(r'(\d)\~(\d)', r'\1-\2', Ctext)
        Ctext = re.sub(r'~', ' tilde ', Ctext)
        Ctext = re.sub(r'\s{2,6}', ' ', Ctext)
        Ctext = re.sub(r'\%\s*\)?', ' percent ', Ctext)


        token_test = tokenizer_bert.tokenize(Ctext)  # 전처리한 소견을 토큰화
        token_test = merge_wordpieces(token_test)  # 토큰 데이터를 재결합
        token_test = reorg_wordpieces(token_test)  # 토큰 데이터의 불용어 제거 및 영문+한글 단어의 정형화

        #matches = re.findall(r'\.|\,', Ctext)
        matches = re.findall(r'[가-힣]', raw_data)
        # if matches:
        #     #print(matches)
        #     print("##########################################")
        #     print(raw_data)
        #     print(f"Need to Replace\n{Ctext}")
        #     ccnt = 1
        #     for t in list(set(token_test)):
        #         if ccnt %10 == 0: print()
        #         print(f"'{t}'", end= ' ')
        #         ccnt += 1
        #     print()
        #     print("##########################################")
            # cnt += 1
            # if cnt == 30 : break

        ## 모든 Raw Data 탐색 결과를 저장 후 return.
        raw_conc.append(raw_data)
        after_conc.append(Ctext)

        redf.loc[i, 'conclusion'] = Ctext

    return raw_conc, after_conc

# 테스트 목적의 csv 파일 반환.
# Raw_Text : 텍스트 데이터 원본.
# After_Text : Preprocessing 결과 텍스트.
def Get_DataFrame_to_CSV(raw_txt, af_txt):
    df = pd.DataFrame({
        'Raw_Text': raw_txt,
        'After_Text': af_txt
    })

    df.to_csv('output.csv', index=False, encoding='utf-8-sig')  # 윈도우에서 한글 포함 시 utf-8-sig 권장


def Acute_Classification(df : pd.DataFrame, redf : pd.DataFrame) :
    for i in range(df.shape[0]):
        row = df.iloc[i]
        Atext = int(str(row['AcuteInfarction']).strip())
        redf.loc[i, 'class'] = Atext


# '##', '-'로 토큰이 분리된 용어를 재결합.
def merge_wordpieces(tokens : list):
    words = []
    current_word = ''
    for token in tokens:
        if token.startswith('##'):
            current_word += token[2:]
        elif token.startswith('-'):
            current_word += token
        elif current_word and current_word[-1] == '-':
            current_word += token
        else:
            if current_word:
                words.append(current_word)
            current_word = token
    if current_word:
        words.append(current_word)
    return words


# 불용어(and, the, 그, at 등) 제거 및 영문+한글('edema로', 'CTA를') 조합에서 불필요한 한글 제거.
def reorg_wordpieces(tokens : list):
    filtered_words = [tk.lower() for tk in tokens if tk.lower() not in stopwords]
    for idx, token in enumerate(filtered_words):
        if re.search(r'[가-힣]', token) and re.search(r'[a-zA-Z-]', token):
            token = re.sub(r'[^a-zA-Z-]', '', token)    # 영어 + 한글 조합의 토큰에서 한글 제거
            filtered_words[idx] = token

        # 한글 표현에서 "[용어]의", "[용어]로" 등의 형태로 토큰화될 수 있는 표현 전처리
        if re.search(r'[가-힣]+([의에은을인이과]|으로)$', token):
            token = re.sub(r'([가-힣]+)(?:[의에은을인이과]|으로)$', r'\1', token)
        elif re.search(r'[a-zA-Z]+[로에은을이]$', token):
            token = re.sub(r'([a-zA-Z]+)[로에은을이]$', r'\1', token)

        # 동일한 의미를 여러 표현으로 나타내는 경우에 대한 정형화.
        token = re.sub(r'않(고|다|으며|는다|음|았음|아)$', r'않다', token)               # 않고, 않으며, 않음 등의 표현을 '않다'로 통일.
        token = re.sub(r'없(고|다|으며|음|었음|어|어보임)$', r'없다', token)             # 없고, 없으며, 없음 등의 표현을 '없다'로 통일.
        token = re.sub(r'있(고|다|으며|음|어)$', r'있다', token)                       # 없고, 없으며, 없음 등의 표현을 '없다'로 통일.
        token = re.sub(r'(관찰|감소|증가|획득|발견|듯|환자).*', r'\1', token)           # 앞의 단어로 의미 설명이 충분한 것들.
        token = re.sub(r'보(임|이다|이며|인다|이는|이고|여)$', r'보이다', token)

        # 대/소문자 구분되는 특정 단어의 정형화
        if token == 'rt':
            token = 'right'
        elif token == 'lt':
            token = 'left'
        elif re.search(r'소견[a-zA-Z가-힣]', token):
            token = '소견'
        elif token == '근위내경돔갱':         token = '근위내경동맥'        #오타 정정
        elif token == '임':                  token = '있다'               # 임 -> 이다 -> 있다
        elif token in ['분', '환자', 'pt']:   token = 'patient'          # 분 (시간 단위는 전처리해서 삭제되었음을 전제로 둠) -> 사람 -> 환자
        filtered_words[idx] = token
    return filtered_words


def sent_tokenizing(df : pd.DataFrame):
    MAX_LEN = 512
    sentences_list = []

    # Findings, Conclusion Tokenizing
    for idx, sents in enumerate(zip(df.finding, df.conclusion)):
        # tokenizer2 = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
        sentences = sents[0] + sents[1]
        sentences_list.append(sentences)        # Findings + Conclusion 텍스트 데이터를 하나의 문자열로 저장.

    #print(sentences_list)
    # tokenized_sentences = []
    # for s in sentences_list :
    #     t = tokenizer_bert.tokenize(s)
    #     tokenized_sentences.append(t[:MAX_LEN])

    # 단어 토큰에 고유한 인덱스 번호를 부여하고, 패딩을 첨가해 시퀀스 생성.
    #input_ids = [tokenizer_bert.convert_tokens_to_ids(x) for x in tokenized_sentences]
    #input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="pre", padding="pre")

        # bert_sentences = '[CLS] '
        # for s in sentences :
        #     bert_sentences += s + ' [SEP] '
    #print(input_ids)
        #token_test = word_tokenize(Ctext)
        # token_test = tokenizer.tokenize(Ctext)          # 전처리한 소견을 토큰화
        # token_test = merge_wordpieces(token_test)       # 토큰 데이터를 재결합
        # token_test = reorg_wordpieces(token_test)       # 토큰 데이터의 불용어 제거 및 영문+한글 단어의 정형화
        # token_list.append(token_test)




if __name__ == '__main__':

    ## 1.Raw Dataset, Raw DataFrame, Preprocessed DataFrame
    kiumSet = pd.read_csv(r'.\TrainCopySet.csv')
    df = pd.DataFrame(kiumSet)
    pre_df = pd.DataFrame(columns=['finding', 'conclusion', 'class'])

    ## 2. Show Raw DataSet(DataFrame) Information.
    show_info(df)
    """
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6190 entries, 0 to 6189
    Data columns (total 3 columns):
     #   Column             Non-Null Count  Dtype 
    ---  ------             --------------  ----- 
     0   Findings           4814 non-null   object
     1   Conclusion         6156 non-null   object
     2   AcuteInfarction    6190 non-null   int64 
    dtypes: int64(1), object(2)
    memory usage: 145.2+ KB
    """

    ## 3.Missing Value Handling
    empty_to_missing(df)

    ## 4.'Findings' Sentence Preprocessing
    raw_find, after_find = Findings_Preprocessing(df, pre_df)


    # 번외. 테스트 목적의 데이터프레임 csv 추출.
    #Get_DataFrame_to_CSV(raw_find, after_find)


    ## 5.'Conclusion' Sentence Preprocessing
    #raw_conc, after_conc = Conclusion_Preprocessing(df, pre_df)

    #Get_DataFrame_to_CSV(raw_conc, after_conc)

    ## 6.Classification ('AcuteInfarction' Preprocessing)
    Acute_Classification(df, pre_df)

    ## 7.DataSet Tokenization
    #sent_tokenizing(pre_df)


    #print(pre_df[0:30]['conclusion'])