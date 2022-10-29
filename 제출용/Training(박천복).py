# Pytorch 기반 모델 학습 코드
# 코드에 대한 설명은 validation.py에 작성했습니다.
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from nltk.tokenize import sent_tokenize # 문장 자연어 토큰화
from tensorflow.keras.preprocessing.sequence import pad_sequences  #Keras 시퀀스
from transformers import BertTokenizer # BERT 토크나이저 모델 활용
from datetime import datetime as dt
import pandas as pd
import numpy as np
import torch, re, nltk, time, random
nltk.download('punkt')     # 구두점 분리가 학습된 모델
nltk.download('stopwords') # 불용어 사전

'''
예시. PATH = '.\TrainSet _1차.csv'
'''
PATH = '.\TrainSet _1차.csv'  # 테스트 파일 경로 기입
MAX_LEN = 512  # BERT 최대 토큰 파라미터

'''
1. 모든 필드의 데이터에 줄넘김 '\n' 문자열이 존재. 이를 띄어쓰기(' ')로 변환한다.
2. Conclusion 필드의 값이 NULL이면 AcuteInfarction(진단 결과)는 모두 0, 검사 내용도 미비 (MRI...)
   -> 해당 데이터는 중요하지 않으니 제외시켜도 괜찮은 부분일까? - 아니면 결과 0처리 단독으로?
3. Findings 필드의 값이 NULL(NaN)이어도 Conclusion 설명이 적혀있으며 검사 결과도 0과 1로 구분된다.
4. Findings와 Conclusion 두 필드 모두 NULL인 경우는 없다.

5. 항목마다 번호 분류가 있다(ex. (1)(2)..., 1.2..., A.B...). 정규표현식 사용해서 삭제처리.
6. 모든 문장 데이터를 소문자 변환 후 처리한다.
'''
def show_info(df : pd.DataFrame):
    print('----------------------------------------\n'\
    '-------@@@@ 원본 데이터 셋 정보 @@@@-------\n'\
    '----------------------------------------')
    df.info()
    print('-----------------------------------------')


'''
###########################################
원본 데이터에서 존재하는 결측값(NaN) 전처리
###########################################
'''
def empty_to_missing(df : pd.DataFrame):
    print(f"# Findings 결측값 = {df['Findings'].isnull().sum()} #")
    print(f"# Conclusion 결측값 = {df['Conclusion'].isnull().sum()} #")

    # 모든 결측값에 빈 문자열 대체
    df.fillna('', inplace=True)

    print("# 결측값(NaN)을 빈 문자열('') 처리... #\n# -- 처리 결과 -- #")
    # 결측치 처리 결과 출력
    print(f"# Findings 결측값 처리 후 = {df['Findings'].isnull().sum()} #")
    print(f"# Conclusion 결측값 처리 후 = {df['Conclusion'].isnull().sum()} #")
    print('-------------------------------------------------------------')


'''
#################################################################
    '\n' 문자를 띄어쓰기로 변환
    '\r' 문자를 삭제
    별도의 특수문자(-, >, <, (, ), [, ], {, }, & 삭제처리.
    항목에 포함된 번호구조 ('1.', '2.', '1)', '2)'...) 삭제처리.
    소수 형식 (1.3, 2.0 등)은 살리기.
##################################################################
'''
def Pretreatment(df : pd.DataFrame):
    for i in range(df.shape[0]):
        row = df.iloc[i]
        Ftext = ' '.join(map(str, row['Findings'].split('\n'))).strip()
        Ftext = Ftext.replace('\r', '')
        Ctext = ' '.join(map(str, row['Conclusion'].split('\n'))).strip()
        Ctext = Ctext.replace('\r', '')

        Ftext = re.sub('[1-9]\.[^0-9]|[1-9][\)\]]|[|[\-\<\>\(\)\:\[\{\}\]\&]', " ", Ftext)
        Ctext = re.sub('[1-9]\.[^0-9]|[1-9][\)\]]|[|[\-\<\>\(\)\:\[\{\}\]\&]', " ", Ctext)

        Atext = int(str(row['AcuteInfarction']).strip())

        # 행 데이터를 전처리한 값들로 수정.
        df.iloc[i] = [Ftext, Ctext, Atext]



'''
#########################################################################################
    BERT 분류 모델의 경우 각 문장의 앞마다 [CLS]를 붙여 문장 시작을 명시.
    문장의 종료는 [SEP]. 
    [CLS]을 인식함으로써 문장의 처음이라 알 수 있게 하고, 
    [SEP]을 인식함으로써 문장의 끝을 알 수 있다. 
#########################################################################################
'''
def sent_tokenizing(DSet : pd.DataFrame):
    #nltk.download('punkt')     # 구두점 분리가 학습된 모델
    #nltk.download('stopwords') # 불용어 사전
    sentences_list = []

    # Findings, Conclusion 데이터를 모두 사용하므로 하나의 구성요소로 만든다.
    for Fs in zip(DSet.Findings, DSet.Conclusion):
        text = Fs[0] + Fs[1]

        # row text를 tokenizing한 문장.
        sentences = sent_tokenize(text)

        #BERT 형식에 맞게 문장 구조화
        Bert_sentences = "[CLS] "
        for s in sentences:
                Bert_sentences += s + " [SEP] "

        # BERT에 적용할 최종 문장 저장
        sentences_list.append(Bert_sentences)

    return sentences_list


'''
###########################################################################
    한글 토크나이징도 무난한 bert-base-multilingual-cased 사전모델 적용.
    BERT 토크나이저 최대 길이가 512이므로 인덱스 제한.
############################################################################
'''
def BERT_Tokenizing_Model(sentences : list):
    # BERT Tokenizer 최대 길이(MAX_LEN) = 512
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
    tokenized_sentences = []

    # BERT에 사용할 문장을 하나씩 뽑아 단어 토큰화
    for s in sentences:
        sent_token = tokenizer.tokenize(s)
        tokenized_sentences.append(sent_token[:MAX_LEN])

    # 단어 토큰에 고유한 인덱스 번호를 부여하고, 패딩을 첨가해 시퀀스 생성.
    # 패딩 값이나 초과되는 토큰 값은 앞에서부터 채우거나 삭제. (pre)
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_sentences]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="pre", padding="pre")

    return input_ids


'''
###################################################
    토큰 시퀀스에 대응하는 Attention Mask 생성
###################################################
'''
def Attention_Masking(ids_list : list):
    attention_masks = []

    # 패딩은 0이므로 조건에 맞으면 1, 아니면 0으로 마스크 생성
    for s in ids_list:
        mask = [int(i > 0) for i in s]
        attention_masks.append(mask)

    return attention_masks



'''
Pytorch 학습 함수
'''
def Training(model, device, train_dataloader):
    start_time = time.time()
    print("## ----- 모델 학습 진행 ----- ##")
    data_len = len(train_dataloader)

    # 랜덤시드. 매번 섞일 때 다른 값으로 안나오게 고장
    seed = 55
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU

    # 옵티마이저
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    # 에폭수
    epochs = 4
    # 총 훈련 스텝 : 배치반복 횟수 * 에폭  = 620
    total_steps = len(train_dataloader) * epochs
    # 스케줄러 생성
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # 그래디언트(기울기) 초기화
    # 가중치 편향에 대해 새로운 기울기 계산
    model.zero_grad()

    # 학습
    for epo_idx in range(0, epochs):
        # 훈련 모드 설정
        model.train()
        # 손실값 초기화
        total_loss = 0

        print("")
        print(f'##----- Epoch {epo_idx+1} / {epochs} -----##')
        print('Training 진행 중...')

        # 데이터로더에서 배치만큼 반복해서 가져옴
        for idx, batches in enumerate(train_dataloader):
            # 정보 출력
            print(f"[{dt.today().strftime('%H:%M:%S')}] {idx + 1}/{data_len}개 데이터 학습 중...")

            # batch 정보를 device에 넣음
            batch = tuple(batch_val.to(device) for batch_val in batches)

            # batch에서 input_ids, attention_mask, label 추출
            input_ids, input_mask, labels = batch

            # (input_ids에 대한 결과 [0일 수치, 1일 수치]로 표현 -> logits[0]
            logits = model(input_ids, token_type_ids=None, attention_mask=input_mask, labels=labels)

            # 변환 결과에 대한 정보값
            results = logits[0]

            # 정보값 추출해서 더함 (loss)
            total_loss += results.item()

            # Backward 수행으로 그래디언트 계산
            results.backward()

            # 그래디언트 클리핑 (기울기 폭주 방지)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # 그래디언트를 통해 가중치 파라미터 업데이트
            optimizer.step()

            # 스케줄러로 학습률 감소
            scheduler.step()

            # 그래디언트 초기화
            model.zero_grad()

        # 평균 로스 계산
        avg_train_loss = total_loss / len(train_dataloader)
        print(f'학습에 대한 평균 손실값 = {avg_train_loss}')
    print(f'## ------ 최종 학습 완료 (학습시간 = [{time.time() - start_time}])------ ##')


#학습 모델 저장
def Save_Model():
    torch.save(model, '.\model_save_CPU.pt')
    torch.save(model.state_dict(), '.\model_dict_save_CPU.pt')
    print('##--- 학습 모델 저장 완료 ---##')



if __name__ == '__main__':
    # Raw Dataset, DataFrame.
    kiumSet = pd.read_csv(PATH)
    df = pd.DataFrame(kiumSet)

    # Conclusion은 \n 문자가 있으므로 이를 처리하고 진행
    # 경우를 고려해 columns 전부 초기화
    df.columns = ['Findings', 'Conclusion', 'AcuteInfarction']

    # Show Dataset Information
    show_info(df)

    # Findings에는 1376개의 NaN(결측치) 데이터 존재.
    # Conclusion에는 34개의 NaN(결측치) 데이터 존재.
    # 결측값을 빈 문자열('')로 변환 처리.
    empty_to_missing(df)

    Pretreatment(df)
    print('##------------ 전처리 완료 ------------##')

    # 문장 전처리
    train_sentences = sent_tokenizing(df)
    print('##------------ 문장 토큰화 완료 ------------##')

    # 문장 토큰을 단어 토큰으로 세분화
    train_inputs = BERT_Tokenizing_Model(train_sentences)
    print('##------------ 단어 토큰화 완료 ------------##')

    # Attention Mask
    train_masks = Attention_Masking(train_inputs)
    print('##------------ 어텐션 마스크 생성 ------------##')

    # 정답지
    train_labels = df['AcuteInfarction'].values

    # CPU를 사용해 학습 진행.
    print('#----- Pytorch Tensor를 이용한 학습 진행 -----#')
    print('#-----      학습에 사용할 CPU 설정...    -----#')
    device = torch.device("cpu")

    # 학습 진행할 train 데이터를 Tensor 변환
    train_inputs = torch.tensor(train_inputs)
    train_labels = torch.tensor(train_labels)
    train_masks = torch.tensor(train_masks)

    batch_size = 32
    '''
    Dataset은 torch.utils.data.Dataset 의 하위 클래스.
    DataLoader는 Dataset을 샘플에 쉽게 접근할 수 있도록 순회 가능한 객체(iterable)로 담는다.
    RandomSampler는 Dataset을 섞음.
    '''
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # 학습에 사용할 사전학습 BERT 모델 {num_labels = 총 사용될 열 (문제 데이터, 정답지) 2개의 열}
    model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)
    model.cpu()

    # 학습 진행
    Training(model, device, train_dataloader)
    Save_Model()