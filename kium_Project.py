import pandas as pd   # 2차원 Vector
import re             # 정규 표현삭
from transformers import BertTokenizer # BERT 토크나이저 모델 활용
import nltk  # 자연어 처리
from nltk.tokenize import sent_tokenize # 문장 자연어 토큰화
from tensorflow.keras.preprocessing.sequence import pad_sequences  #Keras 시퀀스
import torch, gc
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import tensorflow as tf
from sklearn.model_selection import train_test_split
import random
import time
import datetime
import numpy as np    # 계산


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
      print('--------------------------------------------------')


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
      print('--------------------------------------------------')


# '\n' 문자를 띄어쓰기 처리
# '\r' 문자를 삭제
# 별도의 특수문자(-, >, <, (, ) 삭제처리 및 항목 번호구조 ('1.', '2.', '1)', '2)'...) 삭제처리)
# 소수점 구조(1.2)는 남기고, 숫자와 특수문자 조합은 항목으로 판단해 삭제.
def Pretreatment(df : pd.DataFrame):
      for i in range(df.shape[0]):
            row = df.iloc[i]
            Ftext = ' '.join(map(str, row['Findings'].split('\n'))).strip()
            Ftext = Ftext.replace('\r', '')
            Ctext = ' '.join(map(str, row['Conclusion'].split('\n'))).strip()
            Ctext = Ctext.replace('\r', '')

            #Ftext = re.sub('[1-9]\.[^0-9]|[1-9]\)|[\-\<\>\(\)\:]', "", Ftext)
            #Ctext = re.sub('[1-9]\.[^0-9]|[1-9]\)|[\-\<\>\(\)\:]', "", Ctext)
            Ftext = re.sub('[1-9]\.[^0-9]|[1-9][\)\]]|[|[\-\<\>\(\)\:\[\{\}\]\&]', " ", Ftext)
            Ctext = re.sub('[1-9]\.[^0-9]|[1-9][\)\]]|[|[\-\<\>\(\)\:\[\{\}\]\&]', " ", Ctext)

            Atext = int(str(row['AcuteInfarction']).strip())

            # 행 프레임 데이터를 전처리한 값들로 수정.
            df.iloc[i] = [Ftext, Ctext, Atext]

      #print(df)


'''
pd.DataFrame.sample(
    n = 추출할 표본 개수(1~정수)
    frac = 추출할 표본 비율 (위의 n이랑 둘 중 하나만 사용)
    replace = 복원 추출 유무 (True, False)
    weights = 가중치 부여 (column 이름)
    random_state = 난수 발생 초깃값 (재현 가능성을 위한 경우)
    axis = 0:인덱스 기준, 1:column 기준
)
'''
# 원본 데이터프레임의 행을 무작위 섞음.
def Data_Random_Sampling(df : pd.DataFrame):
      # reset_index = 뒤죽박죽된 이전의 인덱스를 초기화 시킴
      df_shuffled = df.sample(frac=1).reset_index(drop=True)

      Acute_DataFrame = df[df['AcuteInfarction']==1]     # 참 판정 레코드 610
      Non_Acute_DataFrame = df[df['AcuteInfarction']==0] # 거짓 판정 레코드 5580

      # 훈련용/테스트/검증으로 사용할 각 판정 결과의 데이터 개수
      Acute_Train_len = int(len(Acute_DataFrame)*0.8)
      Non_Train_len = int(len(Non_Acute_DataFrame)*0.8)
      Acute_val_len = int(len(Acute_DataFrame)*0.1)
      NoN_val_len = int(len(Non_Acute_DataFrame)*0.1)

      # 훈련개수 = 4952,  테스트개수 = 619, 검증개수 = 619
      train = pd.concat([Acute_DataFrame[:Acute_Train_len],\
                         Non_Acute_DataFrame[:Non_Train_len]])

      test = pd.concat([Acute_DataFrame[Acute_Train_len:Acute_Train_len+Acute_val_len],\
                         Non_Acute_DataFrame[Non_Train_len:Non_Train_len+NoN_val_len]])

      validation = pd.concat([Acute_DataFrame[Acute_Train_len + Acute_val_len:], \
                        Non_Acute_DataFrame[Non_Train_len + NoN_val_len:]])

      print(f'뇌졸중 판정 데이터 개수: {len(Acute_DataFrame)}')
      print(f'뇌졸중 아닌 데이터 개수: {len(Non_Acute_DataFrame)}')

      return [train, test, validation]



'''
      BERT 분류 모델의 경우 각 문장의 앞마다 [CLS]를 붙여 문장 시작을 명시.
      문장의 종료는 [SEP]. 
      [CLS]을 인식함으로써 문장의 처음이라 알 수 있게 하고, 
      [SEP]을 인식함으로써 문장의 끝을 알 수 있다. 
      BERT의 pretrain 방법은 [SEP]를 인식하여 두 문장이 이어지는 문장인지, 관련 없는 문장인지 학습하는 것.
'''
def sent_tokenizing(DSet : pd.DataFrame):
      nltk.download('punkt')     # 구두점 분리가 학습된 모델
      nltk.download('stopwords') # 불용어 사전

      sentences_list = []

      # Findings, Conclusion 데이터를 모두 사용하므로 하나의 구성요소로 만든다.
      for idx, Fs in enumerate(zip(DSet.Findings, DSet.Conclusion)):
            text = Fs[0] + Fs[1]
            sentences = sent_tokenize(text) # tex를 tokenizing한 문장.
            Bert_sentences = "[CLS] "
            for s in sentences:
                  Bert_sentences += s + " [SEP] "

            # BERT에 적용할 Train Record 저장
            sentences_list.append(Bert_sentences)

      print('---- BERT 형식의 문장 생성 예시 ----')
      print(sentences_list[100])
      print('--------------------------------')

      return sentences_list

'''
한글 토크나이징도 무난한 bert-base-multilingual-cased 사전모델 적용.
BERT 토크나이저 최대 길이가 512이므로 인덱스 제한.
'''
def BERT_Tokenizing_Model(sentences : list):
      tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
      #tokenizer2 = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
      # BERT Tokenizer 최대 길이 = 512
      MAX_LEN = 512
      tokenized_sentences = []
      for s in sentences:
            t = tokenizer.tokenize(s)
            # MAX_LEN = max(MAX_LEN, len(t))
            tokenized_sentences.append(t[:MAX_LEN])

      # 단어 토큰에 고유한 인덱스 번호를 부여하고, 패딩을 첨가해 시퀀스 생성.
      input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_sentences]
      input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="pre", padding="pre")

      return input_ids

# 문장 구조에 맞는 어텐션 마스크 생성
def Attention_Masking(ids_list : list):
      attention_masks = []
      for seq in ids_list:
            seq_mask = [float(i > 0) for i in seq]
            attention_masks.append(seq_mask)

      return attention_masks


# GPU 또는 CPU 사용 가능한지 테스트
def Checking_cuda():
      if torch.cuda.is_available():
            device = torch.device("cuda")
            print('%d GPU(s) available.' % torch.cuda.device_count())
            print('Can use the GPU:', torch.cuda.get_device_name(0))
      else:
            device = torch.device("cpu")
            print('No GPU available, using the CPU instead.')

      return device


# 정확도 계산 함수
def flat_accuracy(preds, labels):
      pred_flat = np.argmax(preds, axis=1).flatten()
      labels_flat = labels.flatten()

      return np.sum(pred_flat == labels_flat) / len(labels_flat)


# 시간 표시 함수
def format_time(elapsed):
      # 반올림
      elapsed_rounded = int(round((elapsed)))

      # hh:mm:ss으로 형태 변경
      return str(datetime.timedelta(seconds=elapsed_rounded))


'''
Pytorch 학습 함수
'''
def Training(model, device, train_dataloader):
      # GPU 환경변수 설정 (윈도우)
      # set 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'
      # GPU 캐시 초기호
      gc.collect()
      torch.cuda.empty_cache()

      # 옵티마이저
      optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
      # 에폭수
      epochs = 4
      # 총 훈련 스텝 : 배치반복 횟수 * 에폭  = 620
      total_steps = len(train_dataloader) * epochs
      # 스케줄러 생성
      scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
      # 랜덤시드 고정
      seed = 55
      torch.cuda.manual_seed_all(seed)  # GPU 모델 전부
      torch.cuda.manual_seed(seed)
      torch.manual_seed(seed) # CPU?
      random.seed(seed)
      np.random.seed(seed)

      # 그래디언트(기울기) 초기화
      # 가중치 편향에 대해 새로운 기울기 계산
      model.zero_grad()

      # 학습
      for epoch_i in range(0, epochs):
            print("")
            print(f'##----- Epoch {epoch_i+1} / {epochs} -----##')
            print('Training...')

            # 시작 시간
            start_time = time.time()
            # 손실값 초기화
            total_loss = 0
            # 훈련 모드 설정
            model.train()

            # 데이터로더에서 배치만큼 반복해서 가져옴
            for step, batch in enumerate(train_dataloader):
                  # 경과 정보 표시
                  if step % 100 == 0 and not step == 0:
                        elapsed = format_time(time.time() - start_time)
                        print('Batch {:>5,}  of  {:>5,}. Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

                  # 배치를 device에 넣음
                  batch = tuple(t.to(device) for t in batch)

                  # 배치에서 데이터 추출
                  b_input_ids, b_input_mask, b_labels = batch

                  # Forward 수행
                  outputs = model(b_input_ids,
                                  token_type_ids=None,
                                  attention_mask=b_input_mask,
                                  labels=b_labels)

                  # 손실(loss) 구함
                  loss = outputs[0]
                  # 총 로스 계산
                  total_loss += loss.item()
                  # Backward 수행으로 그래디언트 계산
                  loss.backward()

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

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(format_time(time.time() - start_time)))

            # ========================================
            #               Validation
            # ========================================
            print("")
            print("Running Validation...")

            # 시작 시간
            start_time = time.time()
            # 평가 모드
            model.eval()

            # 초기화
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            # 데이터로더에서 배치만큼 반복해서 가져옴
            for batch in validation_dataloader:
                  # 배치를 device에 넣음
                  batch = tuple(t.to(device) for t in batch)

                  # 배치에서 데이터 추출
                  b_input_ids, b_input_mask, b_labels = batch

                  # 그래디언트 계산 안함
                  with torch.no_grad():
                        # Forward 수행
                        outputs = model(b_input_ids,
                                        token_type_ids=None,
                                        attention_mask=b_input_mask)

                  # 로스 구함
                  logits = outputs[0]

                  # CPU로 데이터 이동
                  logits = logits.detach().cpu().numpy()
                  label_ids = b_labels.to('cpu').numpy()

                  # 출력 로짓과 라벨을 비교하여 정확도 계산
                  tmp_eval_accuracy = flat_accuracy(logits, label_ids)
                  eval_accuracy += tmp_eval_accuracy
                  nb_eval_steps += 1

            print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
            print("  Validation took: {:}".format(format_time(time.time() - start_time)))

      print("")
      print("Training complete!")


# 학습 모델 저장
def Save_Model():
      torch.save(model, '.\model_save_CPU.pt')
      torch.save(model.state_dict(), '.\model_dict_save_CPU.pt')
      print('##--- 학습 모델 저장 완료 ---##')


# 검증 말고 테스팅 용도로 따로 사용
def Testing(model, device, test_dataloader):
      # 시작 시간
      start_time = time.time()
      # 평가모드로 변경
      model.eval()
      # 변수 초기화
      eval_loss, eval_accuracy = 0, 0
      nb_eval_steps, nb_eval_examples = 0, 0

      # 데이터로더에서 배치만큼 반복하여 가져옴
      for step, batch in enumerate(test_dataloader):
            # 경과 정보 표시
            if step % 100 == 0 and not step == 0:
                  elapsed = format_time(time.time() - start_time)
                  print('  Batch {:>5,}  of  {:>5,}. Elapsed: {:}.'.format(step, len(test_dataloader), elapsed))

            # 배치를 device에 넣음
            batch = tuple(t.to(device) for t in batch)

            # 배치에서 데이터 추출
            b_input_ids, b_input_mask, b_labels = batch

            # 그래디언트 계산 안함
            with torch.no_grad():
                  # Forward 수행
                  outputs = model(b_input_ids,
                                  token_type_ids=None,
                                  attention_mask=b_input_mask)

            # 로스 구함
            logits = outputs[0]

            # CPU로 데이터 이동
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # 출력 로짓과 라벨을 비교하여 정확도 계산
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

      print("")
      print("Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
      print("Test took: {:}".format(format_time(time.time() - start_time)))


if __name__ == '__main__':
      # Raw Dataset, DataFrame.
      kiumSet = pd.read_csv('.\TrainSet _1차_복사.csv')
      df = pd.DataFrame(kiumSet)
      # Show Dataset Information
      show_info(df)

      # Missing Value Handling
      empty_to_missing(df)

      # Before Pretreatment Text Sample
      sample_str = '''1. <Round 1> Test Sample
      -- 1) Clinical information : Adenocarcinoma of lung(IA)-NSCLC.
      -- [ { } ] if item number is alpha? ><
      -- 3] 28 x 27 x 26 mm and, 5.7 x 1.7 cm. way 2021-10-27  MRA : n/s
      ->- PTO(parietal-temporo-occipital) lobes.'''

      print('##------ 샘플에 포함되는 데이터 전처리 예시 ------##')
      #print(df.iloc[865]['Findings'])
      print(f"샘플 텍스트\n{sample_str}")
      print('----------------------------------------')
      print('##------ 전처리한 샘플 데이터 결과 ------##')
      sample_pret = re.sub('[1-9]\.[^0-9]|[1-9][\)\]]|[|[\-\<\>\(\)\:\[\{\}\]]', " ", sample_str)
      print(sample_pret)
      print('----------------------------------------')

      print('---@@@ 위의 처리처럼 전체 데이터 전처리 진행 @@@---')
      Pretreatment(df)
      print('##------------ 전처리 완료 ------------##')

      # 정제한 데이터를 바탕으로 훈련, 테스트, 검증 데이터셋 생성
      train, test, validation = Data_Random_Sampling(df)

      # 훈련, 테스트, 검증에 사용할 데이터를 BERT에 적용하기 위해 문장 토큰화.
      train_sentences = sent_tokenizing(train)
      test_sentences = sent_tokenizing(test)
      validation_sentences = sent_tokenizing(validation)

      # 정답지
      train_labels = train['AcuteInfarction'].values
      test_labels = test['AcuteInfarction'].values
      validation_labels = validation['AcuteInfarction'].values

      # 문장 토큰을 단어 토큰으로 세분화 --> 단어 시퀀스 생성 (BERT Tokenizing)
      train_inputs = BERT_Tokenizing_Model(train_sentences)
      test_inputs = BERT_Tokenizing_Model(test_sentences)
      validation_inputs = BERT_Tokenizing_Model(validation_sentences)

      #print(train_sentences[3])
      #print(train_token[3])

      # Making Attention Mask
      train_masks = Attention_Masking(train_inputs)
      test_masks = Attention_Masking(test_inputs)
      validation_masks = Attention_Masking(validation_inputs)

      print('#----- Pytorch Tensor를 이용한 학습 진행 -----#')
      print('#----- 학습에 사용할 연산 하드웨어 설정... -----#')
      #device = Checking_cuda()
      device = torch.device("cpu")

      # train, validation 데이터를 파이토치 텐서로 변환 (학습 알고리즘에 사용할 목적)
      train_inputs = torch.tensor(train_inputs)
      train_labels = torch.tensor(train_labels)
      train_masks = torch.tensor(train_masks)
      validation_inputs = torch.tensor(validation_inputs)
      validation_labels = torch.tensor(validation_labels)
      validation_masks = torch.tensor(validation_masks)
      test_inputs = torch.tensor(test_inputs)
      test_labels = torch.tensor(test_labels)
      test_masks = torch.tensor(test_masks)

      print(f"Dimension of tensor: {train_inputs.ndim}")  # ndim(차원) 확인
      print(f"Shape of tensor: {train_inputs.shape}")  # shape(모양) 확인
      print(f"Datatype of tensor: {train_inputs.dtype}")  # 자료형 확인
      print(f"Device tensor is stored on: {train_inputs.device}")  # 어느 장치에 저장되는지 ex) gpu, cpu


      batch_size = 4
      '''
      Dataset은 torch.utils.data.Dataset 의 하위 클래스.
      DataLoader는 Dataset을 샘플에 쉽게 접근할 수 있도록 순회 가능한 객체(iterable)로 감쌉니다.
      RandomSampler는 Dataset을 섞음.
      '''
      train_data = TensorDataset(train_inputs, train_masks, train_labels)
      train_sampler = RandomSampler(train_data)
      train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

      validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
      validation_sampler = SequentialSampler(validation_data)
      validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

      test_data = TensorDataset(test_inputs, test_masks, test_labels)
      test_sampler = RandomSampler(test_data)
      test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

      model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)
      model.cuda()

      Training(model, device, train_dataloader)
      Save_Model()

      '''
      Testing(device, test_dataloader)
      PATH = '.\model_save.pt'
      model = torch.load(PATH)
      model.eval()
      '''