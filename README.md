# CLAMI_set

This is `clami_dir` branch 

**clami와 다른 점** 

- 파일경로로 디렉토리를 입력할 경우, 디렉토리 아래에 있는 모든 파일을 실행시킨다. 
  - 조건 - 모든 arff 파일들이 class attribute name 과 positive label value가 같아야 한다.
- `-t`옵션으로 임계값을 설정할 수 있음 
- 출력형식이 다음과 같다. 
  - 파일 이름,TP,FP,TN,FN,precision,recall,f1,AUC

