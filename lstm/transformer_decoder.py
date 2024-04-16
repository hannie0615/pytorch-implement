import sentencepiece as spm
import pandas as pd
import numpy as np
import csv
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import *

# 1. 토큰화(sentencepiece.unigram)
if not os.path.isfile('ptb.vocab'):
    spm.SentencePieceTrainer.Train('--input=dataset/ptb.train.txt '
                               '--model_prefix=ptb '
                               '--vocab_size=1024 '   # token : 1024
                               '--model_type=unigram '
                               '--max_sentence_length=9999')

# vocab load
vocab_list = pd.read_csv('ptb.vocab', sep='\t', header=None, quoting=csv.QUOTE_NONE)
print(vocab_list.sample(10))
print(vocab_list.shape)     # 68,2

# model load
sp = spm.SentencePieceProcessor()
vocab_file = "ptb.model"
sp.load(vocab_file)


# 2. 데이터 로드
# train file 로드하기
lines = []
f = open('dataset/ptb.train.txt', 'r')
while True:
    line = f.readline()
    if not line:
        break
    lines.append(line)
f.close()

encoded_train = []
for line in lines:
    encoded_train.append(sp.encode_as_ids(line))

sequences = []
for i in range(0, len(encoded_train)):
    for j in range(1, len(encoded_train[i])):
        if len(encoded_train[i]) > 1:
            sequence = encoded_train[i][j-1:j+1]
            sequences.append(sequence)
sequences = np.array(sequences)

from keras.utils import to_categorical
X, y = sequences[:, 0], sequences[:, 1]
y = torch.from_numpy(y)
X = torch.from_numpy(X)
print(f"y.shape : {y.shape}")


# test 데이터셋
path = "dataset/ptb.test.txt"
f = open(path, 'r')
lines = []
while True:
    line = f.readline()
    if not line:
        break
    lines.append(line)
f.close()

testencoded = []
for line in lines:
    testencoded.append(sp.encode_as_ids(line))

sequences = []
for i in range(0, len(testencoded)):
    for j in range(1, len(testencoded[i])):
        if len(testencoded[i]) > 1:
            sequence = testencoded[i][j-1:j+1]
            sequences.append(sequence)
sequences = np.array(sequences)
test_X, test_y = sequences[:, 0], sequences[:, 1]
test_X = torch.from_numpy(test_X)
test_y = torch.from_numpy(test_y)


# 3. 디코더 모델 build, file save
# using pytorch and keras
# LSTM and Transformer
# dim=256, layer=3, transformer head=4
if os.name == 'nt':     # Windows
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
elif os.name == 'posix':        # Mac, Linux
    device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
else:
    print("Unsupported operating")
print(f"device: {device}")

# Model
model = TransformerDecoder(num_tokens=1024, dim_model=256, num_heads=4).to(device)
print(f"model: {model}")

print('Total parameters in model: {:,}'.format(get_total_params(model)))

# Train model
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
total_loss = 0
num_epochs = 10
num_layers = 3
batch_size = 64
hidden_size = 256
num_steps = 20
vocab = 1024


for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for i in range(batch_size*num_steps, X.shape[0]-646, batch_size*num_steps):
        inputs = X[i:i + batch_size*num_steps].to(device)
        targets = y[i:i + batch_size*num_steps].to(device).long()
        inputs = inputs.reshape(batch_size, num_steps)  # X
        targets = targets.reshape(batch_size, num_steps)    # y

        # 이제 tgt를 1만큼 이동하여 <SOS>를 사용하여 pos 1에서 토큰을 예측합니다.
        y_input = y[i-batch_size*num_steps:i].to(device)
        y_input = y_input.reshape(batch_size, num_steps)

        # 다음 단어를 마스킹하려면 마스크 가져오기
        sequence_length = y_input.size(1)
        tgt_mask = model.get_tgt_mask(sequence_length).to(device)

        # X, y_input 및 tgt_mask를 전달하여 표준 training
        pred = model(inputs, y_input, tgt_mask)

        # Permute 를 수행하여 batch size 가 처음이 되도록
        pred = pred.permute(1, 2, 0)
        loss = criterion(pred, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item()

        if i % 1000 == 0:
            print('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
                  .format(epoch + 1, num_epochs, i, X.shape[0], loss.item(), np.exp(loss.item())))

        # Save the model checkpoints
    torch.save(model.state_dict(), "transformer.ckpt")


# Epoch [10/10], Step[1728000/1768326], Loss: 5.5138, Perplexity: 248.10
# Epoch [10/10], Step[1760000/1768326], Loss: 5.2184, Perplexity: 184.64