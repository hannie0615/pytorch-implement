## LSTM Decoder Model - HW01


### python 설정 (mac os 기준)
```
# 1. python.org에서 버전 설치
> which python
> echo ~

# 2. pip 버전 설정
> which pip3    # /usr/local/bin/pip3
> alias pip=/usr/local/bin/pip3
> pip3 install --upgrade pip
```

### 가상환경(venv) 설정
~~~
> pip install sentencepiece
> pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
> pip install pandas numpy tensorflow keras einops
~~~

### 결과
~~~
# LSTM
Total parameters in model: 2,104,320
test loss : 4.652682304382324
test perplexity : 131.58116149902344

# Transformer
Total parameters in model: 6,840,320


~~~