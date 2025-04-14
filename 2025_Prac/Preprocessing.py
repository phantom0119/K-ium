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


# 이미 정형화된 수치 데이터에 대한 마스킹 작업 (중복 변환 방지 목적).
mask_matches = []
def Mask_Repl(match) :
    token = f"__PROTECT{len(mask_matches)}__"
    mask_matches.append((token, match.group(0)))  # 원문 저장
    return token



# Findings 데이터 전처리 작업 ( 학습에 불필요한 단어(용어)를 사전에 제거/변환하므로써 분류 성능을 높일 목적 )
# 사람이 이해하기 쉽도록 구분할 목적의 순서 기호 ( 1., 2. 등)
# 특수 문자 표현 (2개 이상의 줄넘김 또는 --> 등의 방향 표시 등)
def Findings_Preprocessing(df : pd.DataFrame) :
    cnt = 0             # Test print count
    raw_find = []       # Findings Raw Data List
    after_find = []     # Findings Preprocessing List
    for i in range(df.shape[0]) :   # shape는 (Row 수, Column 수)
        row = df.iloc[i]
        Ftext = ' '.join(map(str, row['Findings'].split('\n'))).strip()
        Ftext = Ftext.replace('\r', '')
        raw_data = Ftext

        ##  1. Findings에 포함된 'Clinical information(CI)' Keyword 제거
        #   분류 기준이 아닌 소견 내용 구분 목적의 텍스트이므로 삭제.
        Ftext = re.sub(r'Clinical information\s*:|\*\s*CI\s?:|CI\,', '', Ftext)

        ## 2. 양성과 음성을 구분하는 문자를 명확한 단어로 변경한다.
        ## (+) --> positive, (-) --> negative
        if '(+)' in Ftext :
            Ftext = re.sub(r'\(\+\)|\w\s\+', 'positive', Ftext)

        if '(-)' in Ftext:
            Ftext = re.sub(r'\(\-\)', 'negative', Ftext)

        ##  3. 크기가 변경되는 데이터를 증가(increase) 또는 감소(decrease)로 변경한다.
        #   하나의 의미로 통합하기 위해 특수 문자를 포함한 크기 값의 텍스트를 변경함.
        #   ex) 18mm --> 24mm   = increase
        #   ex) 18 mm --> 9 mm  = decrease
        #   '-->'를 기준으로 이전 값과 이후 값을 추출하여 비교한다.
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

        ##  4. 날짜 기록 데이터 (2011.07.08.), (2011. 11. 11.), (2004) 전체 삭제.
        #   날짜 데이터는 '이전'의 의미를 전달할 뿐, 크게 의미 있지 않다고 판단하여 텍스트 삭제 진행.
        Ftext = re.sub(r'\(?\d{4}\.\d{1,2}\.\d{1,2}\.?\)?|'
                       r'\(\d{4}\)|'
                       r'\(?\d{4}\.\s\d{1,2}\.\s\d{1,2}\.\)', '', Ftext)

        ##  5. 특정 기호나 특수 문자 전체 삭제.
        #   소견 내용을 읽기 쉽게 구분하는 문자( --, -, *, [ 등)는 결과 분류에 필요한 데이터가 아니므로 삭제(띄어쓰기로 텍스트 변환).
        Ftext = re.sub(r'\-\-|\?\)?|\!|\:|\(?\*\)?|\s\-\s|\[', ' ', Ftext)


        ##  6. 순서 번호, 구분 기호를 의미하는 특수 문자 제거.
        #   ex) '1.', '2.', '(1)', '(2)', ' - ', '->, -->', '→', '1)', '[1]', ';' 등
        #   특정 기호들의 조합, 숫자/문자를 결합한 표현은 명확하게 구분하여 제거할 수 있도록 정규표현식 작성에 주의.
        #print(f"Before Text\n{Ftext}")
        Ftext = re.sub(r'([^\W\d_])>|'
                    r'(?<=MRV)\s*>|'
                    r'\s*\-+>|'
                    r'\s*\=+>|'
                    r'\*>|'
                    r'<[a-zA-Z가-힣]|'
                    r'<\-{1,2}|'
                    r';|'
                    r'\d\.(?!\d)\)?|'
                    r'\(\d\)(\:|\.|\,)?|'
                    r'\[\d\]|'
                    r'(\s)\(\*\,|'
                    r'(\s)\.\s|'
                    r'(?<=\w)\.\)?\,|'
                    r'(?<=\w|\))(\]|\))[,.]|'
                    r'(?<=\w)\s{0,1}\,\.?|'
                    r'(?<=[가-힣a-zA-Z])\s?\.{1,2}\)?|'
                    r'\((?=\s{0,1}\w|\<|\>|\#|\=)|'
                    r'(?<=\w|\%)\)|'
                    r'(?<=\w)\]'
                    , r'\1', Ftext, flags=re.VERBOSE)
            #print(f"End conv\n{Ftext}")


        ##  7. 대소 비교 문자('<', '>') 텍스트 변환.
        #   '<', '>'는 다른 특수 문자와 조합하여 '구분 문자' 역할로 사용하는 경우도 있다.
        #   이를 제외(삭제)하면 크기나 백분율 비교로 사용하므로 이들에 대한 명확한 텍스트 변환이 필요하다.
        #   대소 구분 목적으로 사용하는 기호 표현을 'less than', 'greater than'으로 텍스트 변경한다.
        Ftext = re.sub(r'>\s(?=right|left|Lt|Rt|\d|Grade)', 'greater than ', Ftext)
        Ftext = re.sub(r'<\s(?=right|left|Lt|Rt|\d|Grade)', 'less than ', Ftext)
        # Ftext = re.sub(r'<', 'less than', Ftext)

        ## 8. 2회 이상의 띄어쓰기 또는 줄바꿈 문자에 대해 한 번의 줄바꿈만 적용.
        #print(f"Start conv\n{Ftext}")
        Ftext = re.sub(r'\s{2,20}', ' ', Ftext)

        # 테스트 목적의 조건문 탐색
        if ')' in Ftext :
            #or 'greater' in Ftext:
            #print(f"Before Change\n{Ftext}")
            #Ftext = re.sub(r'(\w)(\,|\.)', r'\1', Ftext, flags=re.VERBOSE)
            print(f"After Change\n{Ftext}")

        ## 모든 Raw Data 탐색 결과를 저장 후 return.
        raw_find.append(raw_data)
        after_find.append(Ftext)

        # 테스트 목적의 반복문 도중 반환.
        if cnt == 100 :
            break

    return raw_find, after_find


# Conclusion 데이터 전처리 작업 ( 학습에 불필요한 단어(용어)를 사전에 제거/변환하므로써 분류 성능을 높일 목적 )
def Conclusion_Preprocessing(df : pd.DataFrame) :
    cnt = 0         # Test print count
    raw_conc = []   # Conclusion Raw Data List
    after_conc = [] # Conclusion Preprocesing List

    for i in range(df.shape[0]) :   # shape() = (Row 수, Column 수)
        row = df.iloc[i]
        Ctext = ' '.join(map(str, row['Conclusion'].split('\n'))).strip()
        Ctext = Ctext.replace('\r', '')
        raw_data = Ctext

        ## x. 특수문자가 포함된 의학 용어 변환.
        Ctext = re.sub(r'[pP]\-[cC][Oo][Mm]\.?\s*a((\.|\))\)?|rtery)', 'posterior communicating artery ', Ctext)
        Ctext = re.sub(r'[pP]\-[cC][Oo][Mm]|PCOM', 'posterior communicating ', Ctext)
        Ctext = re.sub(r'op\.bed[., (]', 'operative bed ', Ctext)
        Ctext = re.sub(r'[Ff][./-][Uu]|follow up|follow\-up', 'follow-up ', Ctext)
        Ctext = re.sub(r'[Nn][./-][sS]|[nN]on?( other)? [sS]ignificant|without significant change|[nN]o evidence of significant', 'non-specific ', Ctext)
        Ctext = re.sub(r'op\.site[., (]', 'operative site ', Ctext)
        Ctext = re.sub(r'postop\.change[., (]', 'postoperative-change ', Ctext)
        Ctext = re.sub(r'e\.g\.?', ' ', Ctext)
        Ctext = re.sub(r'[sS]\/[pP]', 'status-post ', Ctext)
        Ctext = re.sub(r'[rR][/][oO]', 'rule-out', Ctext)
        Ctext = re.sub(r'[cC][/][Ww]', 'consistent-with', Ctext)




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
                                   fr' Length {Ltmp}mm ',
                                   Ctext)
                    # Width 정형화
                    Ctext = re.sub(fr'(?<=Length {Ltmp}mm ).+{Wvalue}\s*(cm)?(?=(x|\*|X)\.?\s*{Hvalue})',
                                   fr'Width {Wtmp}mm '
                                   , Ctext)
                    # Height 정형화
                    Ctext = re.sub(fr'(?<=Length {Ltmp}mm Width {Wtmp}mm ).+{Hvalue}\s*cm',
                                   fr'Height {Htmp}mm '
                                   , Ctext)
                else :
                    Ctext = re.sub(fr'{grplist[0]}'+r'(?=\s*(mm)?(x|\*|X)\.?\s*(\d{1,2}(\.\d{1,2})?)\s*(mm)?(x|\*|X)\.?\s*(\d{1,2}(\.\d{1,2})?)\s*mm)',
                                   fr' Length {grplist[0]}mm ',
                                   Ctext)
                    Ctext = re.sub(fr'(?<=Length {grplist[0]}mm ).+{grplist[2]}\s*(mm)?(?=(x|\*|X)\.?\s*{grplist[4]})',
                                   fr'Width {grplist[2]}mm ',
                                   Ctext)
                    Ctext = re.sub(fr'(?<=Length {grplist[0]}mm Width {grplist[2]}mm ).+{grplist[4]}\s*mm',
                                   fr'Height {grplist[4]}mm ',
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
                                fr' Length {Ltmp}mm ',
                                Ctext)

                # Width 정형화
                Ctext = re.sub(fr'(?<=Length {Ltmp}mm).+{Wvalue}(?=\s*(x|\*|X)\s*{Hvalue}' + r'\s*\-{1,4}\>?)',
                               fr' Width {Wtmp}mm ',
                               Ctext)

                # Height 정형화
                Ctext = re.sub(fr'(?<=Length {Ltmp}mm Width {Wtmp}mm).+{Hvalue}' + r'\s*\-{1,4}\>?',
                               fr' Height {Htmp}mm change ',
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
                                    fr' Length {Ltmp}mm '
                                    , Ctext)
                    # Width 정형화
                    Ctext = re.sub( fr'(?<=Length {Ltmp}mm ).+{Wvalue}\s*\-?(cm|\))',
                                    fr'Width {Wtmp}mm '
                                    , Ctext)
                else :
                    # mm 단위인데 실수 형태인 경우, 반올림하여 정수화.
                    if grplist[-1] == 'mm' and '.' in grplist[0]:
                        Ltmp = str(round(float(grplist[0])))
                        Lvalue = re.sub(r'\.', r'\\.', grplist[0])
                        Ctext = re.sub(fr'([^1-9]|^){Lvalue}' + r'(?=\s*(mm)?(x|\*|X)\.?\s*(\d{1,2}(\.\d{1,2})?)\s*\-?mm)',
                                       fr' Length {Ltmp}mm ',
                                       Ctext)
                    else :
                        # Length 정형화
                        Ctext = re.sub(
                            fr'([^1-9]|^){grplist[0]}' + r'(?=\s*(mm)?(x|\*|X)\.?\s*(\d{1,2}(\.\d{1,2})?)\s*\-?mm)',
                            fr' Length {grplist[0]}mm '
                            , Ctext)

                    if grplist[-1] == 'mm' and '.' in grplist[2]:
                        Wtmp = str(round(float(grplist[2])))
                        Wvalue = re.sub(r'\.', r'\\.', grplist[2])
                        Ctext = re.sub(fr'(?<=Length {Ltmp}mm ).+{Wvalue}\s*\-?mm',
                                       fr'Width {Wtmp}mm '
                                       , Ctext)
                    else :
                        # Width 정형화
                        Ctext = re.sub(fr'(?<=Length {grplist[0]}mm ).+{grplist[2]}\s*\-?mm',
                                       fr'Width {grplist[2]}mm '
                                       , Ctext)
            # print(matches)
            # print(Ctext)

        # 2차원. 변경된 크기 이전의 값을 의미하는 부분의 정형화 (단위 표시가 없으며 뒤에 '-' 또는 '->', '-->', '--->' 등이 붙는다).
        # 실수로 표현된 값은 'cm' 단위이므로 'mm' 단위로 변환한다.
        matches = re.findall(r'(\d{1,2}(?:\.\d{1,2})?)\s*(x|\*|X)\s*(\d{1,2}(?:\.\d{1,2})?)\s*(\-{1,4}\>?)', Ctext)
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
                else :
                    Wtmp, Wvalue = grplist[2], grplist[2]

                # Length 정형화
                Ctext = re.sub(fr'([^1-9]|^|\(){Lvalue}'+ r'(?=\s*(x|\*|X)\s*\d{1,2}(\.\d{1,2})?\s*\-{1,4}\>?)',
                               fr' Length {Ltmp}mm ',
                               Ctext)

                # Width 정형화
                Ctext = re.sub(fr'(?<=Length {Ltmp}mm ).+{Wvalue}' + r'\s*\-{1,4}\>?',
                               fr'Width {Wtmp}mm change ',
                               Ctext)
            # print(matches)
            # print(Ctext)

        # 1차원 크기 데이터 정형화 (3차원, 2차원 크기 데이터에 대해 모두 정형화가 완료된 상태여야 한다.)
        # 이미 정형화가 완료된 텍스트는 중복 변환되지 않도록 마스킹 처리 후 마지막에 복원하는 방법으로 구현.
        global mask_matches
        mask_matches = []
        mask_pattern = re.compile(r'\b(?:Length|Width|Height)\s+\d{1,2}(?:\.\d{1,2})?mm\b')
        masked = mask_pattern.sub(Mask_Repl, Ctext) # 정형화 전 이미 정형화된 데이터는 겹치지 않도록 마스킹.

        # # 마스킹된 텍스트에서 크기 변경 전 1차원 크기 데이터 추출 (ex. 1.5 - 2.1cm 형태에서 1.5 값)
        matches = re.findall(r'(\d{1,2}(?:\.\d{1,2})?)\s*(\-{1,5}>?)[ 0-9.]*?(cm|mm)', masked)
        if matches :
            # 중복되는 Group이 2번 이상 정형화되지 않도록 세트화.
            matches = list(set(matches))
            for grplist in matches :
                # 2.2-cm, 20-mm 등의 단일 값 처리
                if grplist[1] == '-':
                    if grplist[-1] == 'cm':
                        Ltmp = str(int(float(grplist[0]) * 10))
                        Lvalue = re.sub(r'\.', r'\\.', grplist[0])
                        masked = re.sub(fr'{Lvalue}\-cm', fr' Length {Ltmp}mm ', masked)
                    elif grplist[-1] == 'mm':
                        if '.' in grplist[0]:
                            Ltmp = str(int(round(float(grplist[0]))))
                            Lvalue = re.sub(r'\.', r'\\.', grplist[0])
                            masked = re.sub(fr'{Lvalue}\-mm', fr' Length {Ltmp}mm ', masked)
                        else :
                            masked = re.sub(fr'{grplist[0]}\-mm[.,]?', fr' Length {grplist[0]}mm ', masked)
                # 크기 변동 이전의 값 처리
                elif '>' in grplist[1]:
                    if grplist[-1] == 'cm':
                        Ltmp = str(int(float(grplist[0]) * 10))
                        Lvalue = re.sub(r'\.', r'\\.', grplist[0])
                        masked = re.sub(fr'{Lvalue}\s*{grplist[1]}', fr' Length {Ltmp}mm change', masked)
                    elif grplist[-1] == 'mm':
                        if '.' in grplist[0]:
                            Ltmp = str(int(round(float(grplist[0]))))
                            Lvalue = re.sub(r'\.', r'\\.', grplist[0])
                            masked = re.sub(fr'{Lvalue}\s*{grplist[1]}', fr' Length {Ltmp}mm change', masked)
                        else :
                            masked = re.sub(fr'{grplist[0]}\s*{grplist[1]}', fr' Length {grplist[0]}mm change', masked)

                # 마스킹된 텍스트를 복원 후, 최종 결과를 Ctext에 저장.
                for token, text in mask_matches:
                    masked = masked.replace(token, text)

                Ctext = masked
                #print(grplist)
                # print(Ctext)


        mask_matches = []
        mask_pattern = re.compile(r'\b(?:Length|Width|Height)\s+\d{1,2}(?:\.\d{1,2})?mm\b')
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
                    masked = re.sub(fr'([^1-9]|^|\(){Lvalue}\s*cm', fr' Length {Ltmp}mm ', masked)
                elif grplist[-1] == 'mm' and  '.' in grplist[0] :
                    #Ltmp = re.sub(r'(\d{1,2}).+', r'\1', grplist[0])
                    Ltmp = str(round(float(grplist[0])))
                    Lvalue = re.sub(r'\.', r'\\.', grplist[0])
                    #print(Ctext)
                    masked = re.sub(fr'([^1-9]|^|\(|(?<!Length ))({Lvalue}\s*mm)', fr' Length {Ltmp}mm ', masked)
                    #print(Ctext)
                else :
                    masked = re.sub(fr'([^1-9]|^|\(|(?<!Length )){grplist[0]}\s*mm', fr' Length {grplist[0]}mm', masked)

                # 마스킹된 텍스트를 복원 후, 최종 결과를 Ctext에 저장.
                for token, text in mask_matches:
                    masked = masked.replace(token, text)

                Ctext = masked

        #print(Ctext)

        # 실수 표현된 텍스트 데이터 찾아보기
        # matches = re.findall(r'\d{1,2}\.\d{1,2}', Ctext)
        # if matches and len(matches) >= 2:
        #     print(matches)
        #     print(Ctext)


        ##  2. 날짜 기록 데이터 삭제
        #   '이전'의 의미를 포함할 뿐 날짜 데이터 자체가 판독 결과에 큰 영향을 주지 않으므로 삭제.
        Ctext = re.sub(r'on \d{4}(\.|\-)\d{1,2}(\.|\-)\d{1,2}\,?|'
                        r'in\s\d{4}(\.\d{1,2}\.\d{1,2})?|'
                        r'\(20\d{2}(?=\w)|'
                        r'\(\d{4}\.\d{1,2}\)?\.(\d{1,2}\)(\,|\.))|'
                        r'\(\d{4}\.\d{1,2}\.\s|'
                        r'\(\d{2,4}\.\s?\d{1,2}\.\s?\d{1,2}\.?(\)|\,)(\.|\,)?|'
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
        Ctext = re.sub(r'\((\d|Rt|Lt|\s).*?(cm|mm)\)\.?|'
                       r'(\(|\[).*?(IDX|Img|IM|Idx).*(\)|\])|'
                       r'\(20\d{2}(\.|\-).*\)\.?|'
                       r'DDx\.\)?|'
                       r'rec\)', ' ', Ctext)

        ##  1. Conclusion에 포함된 순서 번호 기호 삭제
        #   순서 번호는 가독성 개선을 목적이지 판독 결과에 영항을 주지 않는다.
        Ctext = re.sub(r'\d(\.|\))\)?(?=\s|[a-zA-Z])', ' ', Ctext)


        ##  3. 불필요한 특수 문자 조합 제거.
        matches = re.findall(r'\;|\:|\s{2,3}\-\-?\>?|\[|\]|'
                       r'(?<=[a-zA-Z가-힣1-9 ])\s*(\(|\))\s*(\.|\,|\)){0,2}(?=[a-zA-Z가-힣1-9<> ]|$)|'
                       r'(\-|\=){1,4}>|\s(\-|\=){2,4}(\s|\w)|'
                       r'\(\=|\&|'
                       r'(?<=\w)\.\s|'
                       r'\d{1,2}\.\d{1,2}\-m{1}|'
                       r'\s(\*|\=)\s|'
                       r'\s+-+(\s+|$)|'
                       r'\s{1,5}\-+(?=\w)|'
                       r'\*{2,3}|^\*|'
                       r' - ',
                    Ctext)


        Ctext = re.sub(r'\;|\:|\s{2,3}\-\-?\>?|\[|\]|'
                       r'(?<=[a-zA-Z가-힣1-9 ])\s*(\(|\))\s*(\.|\,|\)){0,2}(?=[a-zA-Z가-힣1-9<> ]|$)|'
                       r'(\-|\=){1,4}>|\s(\-|\=){2,4}(\s|\w)|'
                       r'\(\=|\&|'
                       r'(?<=\w)\.\s|'
                       r'\d{1,2}\.\d{1,2}\-m{1}|'
                       r'\s(\*|\=)\s|'
                       r'\s+-+(\s+|$)|'
                       r'\s{1,5}\-+(?=\w)|'
                       r'\*{2,3}|^\*|'
                       , ' ', Ctext)


        Ctext = re.sub(r' - ', ' ', Ctext)
        matches = re.findall(r'\s+-\s+', Ctext)
        if matches:
            print("##########################################")
            print(raw_data)
            print(f"Need to Replace\n{Ctext}")
            print("##########################################")
            # cnt += 1
            # if cnt == 30 : break

        ## 모든 Raw Data 탐색 결과를 저장 후 return.
        raw_conc.append(raw_data)
        after_conc.append(Ctext)

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


if __name__ == '__main__':
    # Raw Dataset, DataFrame.
    kiumSet = pd.read_csv(r'.\TrainCopySet.csv')
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
    #raw_find, after_find = Findings_Preprocessing(df)


    # 번외. 테스트 목적의 데이터프레임 csv 추출.
    #Get_DataFrame_to_CSV(raw_find, after_find)


    # 4. 'Conclusion' Sentence Preprocessing
    raw_find, after_find = Conclusion_Preprocessing(df)

    #Get_DataFrame_to_CSV(raw_find, after_find)