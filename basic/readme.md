
## Assginment 01

- 수업 : 기계학습 2024-1
- 이름 : 신한이
- 학번 : 2024021072
- 목표 : 연속형 랜덤 변수 x에 대해 타겟 변수 t를 예측
  -  MLE, MAP, 베이지안 메소드 방법론을 이용

  
 --- 
---
### venv fnfo
~~~
# python 3.12.2
> import numpy matplotlib
~~~
  
---
### 파일 구조
|--- generate_data.py       `````# 데이터셋(x, t) 생성  `````  
|--- poly_mle.py  
|--- poly_map.py  
|--- poly_bayesian.py
    
---
### 실행 방법
~~~
# 1. generate dataset
> python generate_data.py

# 2. regression (순서 상관 없음)
> python poly_mle.py
> python poly_map.py
> python poly_bayesian.py
~~~
  
---
### 예상 결과 화면
![mleplot](https://github.com/hannie0615/pytorch-implement/assets/50253860/af255bae-aaf2-4ed7-8cf2-a82b8076ddf6)
![mapplot](https://github.com/hannie0615/pytorch-implement/assets/50253860/4142041a-6193-4b0e-b83c-57ccbb138f68)
![bayesian](https://github.com/hannie0615/pytorch-implement/assets/50253860/e326d21b-9923-4d4f-98d4-93d49f12d1d6)

