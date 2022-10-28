'''
 # 모델 검증 수행 코드 #
########################################################################################
-  44번 줄의 PATH 변수에 검증에 사용할 데이터 파일(csv) 경로를 입력합니다.
-  검증에 사용할 모델(model_save_CPU.pt)이 코드 파일과 같은 경로에 있는지 확인합니다.
-  코드를 실행하면 검증을 시작합니다.
########################################################################################

%% 개발 환경에서 데이터 셋을 사용한 내용 %%
-------------------------------------------------------------------------------------------
 2차 제출을 가정으로 30% 데이터 개수(2653)를 돌리면 코드에서 83개의 batch Set이 됩니다. 
- GPU(1650 Ti)에서 MAX_LEN=512로 테스트 했을 때, 30% 데이터셋이 8.8분 걸립니다.
-------------------------------------------------------------------------------------------

사용한 환경 및 라이브러리 버전 정보
conda = 22.9.0 (python 3.8.5)
torch = 1.12.1
tensorflow(keras) = 2.10.0
transformers = 4.22.2
scikit-learn = 1.1.1
numpy = 1.23.1
nltk = 3.7
pandas = 1.5.0
matplotlib = 3.5.2
'''
from sklearn.metrics import roc_curve, roc_auc_score  # AUROC
from nltk.tokenize import sent_tokenize # 문장 자연어 토큰화
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import BertTokenizer # BERT 토크나이저 모델 활용
from tensorflow.keras.preprocessing.sequence import pad_sequences  #Keras 시퀀스
from datetime import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch, re, nltk, time

nltk.download('punkt')     # 구두점 분리가 학습된 모델
nltk.download('stopwords') # 불용어 사전

'''
validation.py 저장 위치와 동일한 곳에 csv 파일이 있을 경우
예시. PATH = '.\TrainSet _1차.csv'
'''
PATH = '.\Dataset.csv'  # 테스트 파일 경로 기입
MPATH = '.\model_save_CPU.pt'  # 사용할 모델 경로 (같은 디렉터리에 있다고 가정)
# BERT Tokenizer 최대 길이(MAX_LEN) = 512
MAX_LEN = 512

# 개발에 참고한 URL
# https://velog.io/@seolini43/%EC%9D%BC%EC%83%81%EC%97%B0%EC%95%A0-%EC%A3%BC%EC%A0%9C%EC%9D%98-%ED%95%9C%EA%B5%AD%EC%96%B4-%EB%8C%80%ED%99%94-BERT%EB%A1%9C-%EC%9D%B4%EC%A7%84-%EB%B6%84%EB%A5%98-%EB%AA%A8%EB%8D%B8-%EB%A7%8C%EB%93%A4%EA%B8%B0%ED%8C%8C%EC%9D%B4%EC%8D%ACColab-%EC%BD%94%EB%93%9C
# https://tutorials.pytorch.kr/?_gl=1*qivl2j*_ga*MTI3NTMyNTQ1OS4xNjY2ODg5NjQy*_ga_LEHG248408*MTY2Njg5NTI0NC4xLjAuMTY2Njg5NTI0NC42MC4wLjA.*_ga_L5NC8SBFPY*MTY2Njg5NTI0NS4xLjAuMTY2Njg5NTI0NS42MC4wLjA.*_ga_LZRD6GXDLF*MTY2Njg5NTIzNC4yLjEuMTY2Njg5NTI0NS40OS4wLjA.

'''
##################################
형식에 맞게 변환된 입력 test_dataloader
AUROC 방식으로 테스트 진행
##################################
'''
def Testing(model, device, test_dataloader):
    start_time = time.time()
    print("## ----- 모델 검증 진행 ----- ##")
    data_len = len(test_dataloader)
    
    model.eval()
    # pred_list: 예측결과, label_list: 정답지
    pred_list = []; label_list = []
    
    for idx, batches in enumerate(test_dataloader):
        print(f"[{dt.today().strftime('%H:%M:%S')}] {idx+1}/{data_len}개 데이터Set 입력 완료 중...")

        # batch 정보를 device에 넣음
        batch = tuple(batch_val.to(device) for batch_val in batches)
        # batch에서 input_ids, attention_mask, label 추출
        input_ids, input_mask, labels = batch
        
        # Grad 계산 안함
        with torch.no_grad():
            #(input_ids에 대한 예측 결과 [0일 수치, 1일 수치]로 표현 -> logits[0]
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

    # 예측값(pred_list)과 정답지(label_list)를 AUROC 계산
    fpr, tpr, thresholds = roc_curve(pred_list, label_list)
    # ROC 곡선
    plt.plot(fpr, tpr)
    plt.xlabel('FP Rate')
    plt.ylabel('TP Rate')
    plt.show()

    # AUC 면적
    print(f"정확도(Accuracy): {round(roc_auc_score(pred_list, label_list),6)}")
    print(f"Test 수행시간(Took): {time.time() - start_time}")



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
######################################
    CPU/GPU 사용 가능한 것 선택
######################################
'''
def Checking_cuda():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = torch.load(MPATH, map_location=f"cuda:{torch.cuda.current_device()}")
        model.to(device)  # Torch Tensor  -->  CUDA Tensor 변환
        print(f'# Can use the GPU: {torch.cuda.get_device_name(0)} #')
    else:
        device = torch.device("cpu")
        model = torch.load(MPATH)
        print('# No GPU available, using the CPU. #')

    return device, model



if __name__ == '__main__':
    settime = time().time()
    # 검증에 사용할 데이터를 엑셀(csv)에서 가져온다고 가정
    kiumSet = pd.read_csv(PATH)
    df = pd.DataFrame(kiumSet)

    # Conclusion은 \n 문자가 있으므로 이를 처리하고 진행
    # 혹시 모르니 columns 전부 명시적 변경
    df.columns = ['Findings', 'Conclusion', 'AcuteInfarction']

    # 1. 결측값(NaN) 전처리
    empty_to_missing(df)

    # 2. 불필요한 텍스트 데이터 전처리
    Pretreatment(df)

    # 3. 정제한 데이터를 BERT에 적용하기 위한 문장 토큰화
    test_sentences = sent_tokenizing(df)

    # 4. 문장 토큰을 단어 토큰으로 세분화 --> 단어 시퀀스 생성 (BERT Tokenizing)
    test_inputs = BERT_Tokenizing_Model(test_sentences)

    # Attention Mask
    test_masks = Attention_Masking(test_inputs)

    # 정답지
    test_labels = df['AcuteInfarction'].values


    # 5. 검증에 사용할 device와 model 구조 설정
    '''
        학습된 모델은 cpu로 기반으로 만들었기 때문에 GPU를 사용해 검증할 경우 변환 필요.
        Checking_cuda() 함수에서 CPU, GPU 환경에 맞게 변환 지시.
        CPU로 직접 명시해서 사용하려면 아래의 주석 코드 활용.
    '''
    print('# -------- Pytorch Tensor를 이용한 검증 진행   --------#')
    print('# -------- 검증에 사용할 연산 하드웨어 설정... --------#')
    device, model = Checking_cuda()
    print('-------------------------------------------------------------')

    # GPU 적용이 안 되는 전제가 있다면 아래 주석 코드 사용 (직접 명시)
    # device = torch.device("cpu")


    # 6. 파이토치로 검증하기 위한 텐서 설정
    test_inputs = torch.tensor(test_inputs)
    test_masks = torch.tensor(test_masks)
    test_labels = torch.tensor(test_labels)

    
    # 7. 텐서 Dataset 설정
    '''
      Dataset은 torch.utils.data.Dataset 의 하위 클래스.
      DataLoader는 Dataset을 샘플에 쉽게 접근할 수 있도록 순회 가능한 객체(iterable)로 담는다.
      RandomSampler는 Dataset을 섞음.
    '''
    batch_size = 32  # batch
    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

   

    # 8. 모델 테스팅 진행
    Testing(model, device, test_dataloader)

    print('# 검증 종료 #')
    print(f'# 총 검증 시간 = {time.time() - settime}')