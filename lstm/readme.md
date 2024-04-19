## Decoder Model - HW01


### 가상환경(venv) 설정
~~~
> pip install sentencepiece
> pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
> pip install pandas numpy tensorflow keras einops
~~~
  
---

### 실행 순서
~~~
# 1. vocab 생성
> python tokenizer.py

# 2. model train, save, evaluation
> python lstm_main.py   # lstm
> python transformer_main.py    # transformer
~~~


  
---

### 작성 과정

lstm에서는 learning rate를 기본값(1e-3)에서 올렸을 때
손실이 더욱 커져 기본값을 유지하고  
transformer에서는 학습에 정체가 있어 기본값에서 0.01로 올려주었습니다.



### 결과(Output)
~~~
# LSTM
Total parameters in model: 2,104,320
test loss : 4.652682304382324
test perplexity : 131.58116149902344

# Transformer
Total parameters in model: 6,840,320
test loss : 5.223451137542725
test perplexity : 200.17913818359375

~~~
