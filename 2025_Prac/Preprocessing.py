"""
- TrainCopySet.csv : 원본 데이터 Set
-
"""
import pandas as pd     # DataFrame
import re               # Regular Expression



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


# 데이터 전처리 작업 ( 학습에 불필요한 단어(용어)를 사전에 제거/변환하므로써 분류 성능을 높일 목적 )
# 사람이 이해하기 쉽도록 구분할 목적의 순서 기호 ( 1., 2. 등)
# 특수 문자 표현 (2개 이상의 줄넘김 또는 --> 등의 방향 표시 등)
def Findings_Preprocessing(df : pd.DataFrame) :
    cnt = 0
    for i in range(df.shape[0]) :   # shape는 (Row 수, Column 수)
        row = df.iloc[i]
        Ftext = ' '.join(map(str, row['Findings'].split('\n'))).strip()
        Ftext = Ftext.replace('\r', '')

        ## 1. Findings에 포함된 'Clinical information :' Keyword 제거
        Ftext = re.sub(r'Clinical information\s*:', '', Ftext)

        ## 2. 양성과 음성을 구분하는 문자를 명확한 단어로 변경한다.
        ## (+) --> positive, (-) --> negative
        if '(+)' in Ftext :
            #print(Ftext)
            Ftext = re.sub(r'\(\+\)|\w\s\+', 'positive', Ftext)
            #print(Ftext)

        if '(-)' in Ftext:
            #print(Ftext)
            Ftext = re.sub(r'\(\-\)', 'negative', Ftext)
            #print(Ftext)

        ## 3. 크기가 변경되는 데이터를 증가(increase) 또는 감소(decrease)로 변경한다.
        ## ex) 18mm --> 24mm   = increase
        ## ex) 18 mm --> 9 mm  = decrease
        bef_matches = re.findall(r'(\d+(\.\d+)?)(?=\s*(m|c)m\.?\s*\-+>\s*\d+(\.\d+)?\s*(m|c)m)\.?', Ftext)   # 이전 크기 추출
        # if bef_matches :
        #     print(Ftext)
        #     print(bef_matches)


        af_matches = re.findall(r'(?:\-+>)\s*(\d+(\.\d+)?)(?=\s*(m|c)m\.?)', Ftext) # 이후 크기 추출
        # if af_matches :
        #     print(f"After Test\n{Ftext}")
        #     print(af_matches)

        size_matches = tuple()
        for i in range(len(bef_matches)) :
            bef = bef_matches[i][0]
            af  = af_matches[i][0]
            sent = ""
            #print(f"text bef and af = {bef}, {af}")

            # 크기 변경 정도에 따라 decrease, increase, NoChange 구분.
            if float(bef) > float(af) :
                sent = "decrease"
            elif float(bef) < float(af) :
                sent = "increase"
            elif float(bef) == float(af) :
                sent = "NoChange"

            #print(Ftext)
            # 전체 문자열을 'decrease|increase|NoChange' 중 하나로 변경.
            Ftext = re.sub(str(bef)+r'(\.\d+)?\s*(m|c)m\.?\s*\-+>\s*'+str(af)+r'(\.\d+)?\s*(m|c)m\.?', sent, Ftext)
            #print(Ftext)


        ## 4. 순서 번호, 구분 기호를 의미하는 특수 문자 제거.
        ## ex) '1.', '2.', '(1)', '(2)', ' - ', '->, -->', '→', '1)'  등
        match = re.findall(r'\d\.(?!\d)|\(\d\)|\s\-\s|\-+>|→|\d\)\s|(?<=\w|[가-힣])\s*[)>:]|[(<](?=\w|[가-힣]|\<)|\s*\-[->]', Ftext)
        if match :
            #print(f"Start conv\n{Ftext}")
            Ftext = re.sub(r'\d\.(?!\d)|\(\d\)|\s\-\s|\-+>|→|\d\)\s|(?<=\w|[가-힣])\s*[)>:]|[(<](?=\w|[가-힣]|\<)|\s*\-[->]', '', Ftext)
            #print(f"End conv\n{Ftext}")
            cnt += 1


        ## 2회 이상의 띄어쓰기 또는 줄바꿈 문자에 대해 한 번의 줄바꿈만 적용.
        print(f"Start conv\n{Ftext}")
        Ftext = re.sub(r'\s{2,20}', ' ', Ftext)
        print(f"End conv\n{Ftext}")


        if cnt == 100 :
            break


if __name__ == '__main__':
    # Raw Dataset, DataFrame.
    kiumSet = pd.read_csv('.\TrainCopySet.csv')
    df = pd.DataFrame(kiumSet)

    # 1. Show Raw DataSet(DataFrame) Information.
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

    # 2. Missing Value Handling
    empty_to_missing(df)


    # 3. 'Findings' Sentence Preprocessing
    Findings_Preprocessing(df)