## 2차 제출 파일 목록 ##
1. README2차_박천복.txt
2. Validation박천복.py
3. Validation2차_박천복.py        -> 검증에 사용할 파일
4. model_save_CPU.pt
5. K-ium 기술문서 박천복.pdf 


-- 2차 제출 목록 중 1차 제출에 포함되었던 파일 --
2. Validation박천복.py  = Validation(박천복).py
4. model_save_CPU.pt

2번 파일은 이름만 변경했습니다. 코드 부분은 동일합니다.
-> "Validation2차_박천복.py"에서 import 하기 위한 목적입니다.

4번 파일은 학습 모델이며 검증 과정에서 변경 없이 사용했습니다.


## 출력값 설명 ##
가이드에서 "허혈성 뇌경색이 있을 경우의 모델 출력값 a와 
허혈성 뇌경색이 없을 경우의 모델 출력값 b"를 출력하라고 했습니다.
출력은 1칸 공백을 두어 "a b"로 출력됩니다.

ex.
뇌경색으로 판단한 수치(a) 뇌경색이 아니라고 판단한 수치(b)
0.001 0.999
0.0011 0.9989
0.9783 0.0217
0.0013 0.9987
0.0013 0.9987

3번 행만 뇌경색(1)으로 판단한 것이며 나머지는 뇌경색이 아니라고(0) 판단한 것입니다.

검증 데이터 정확도 : 0.97121

