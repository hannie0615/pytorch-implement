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
"""
input : 학습시킬 파일
model_prefix : 만들어질 모델 이름
vocab_size : 단어 집합의 크기
model_type : 사용할 모델 (unigram(default), bpe, char, word)
max_sentence_length: 문장의 최대 길이
"""
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

model = DecoderRNN(vocab_size=1024, embed_size=256, hidden_size=256, num_layers=3).to(device)
print(model)


# 4. Training
# lr = 1e-3
criterion = F.cross_entropy
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
model.train()

# input = torch.Tensor([7.0, 5.0])
# outputs = model(input)
# loss = criterion(outputs, y)
# print(loss)

# hyper parameters
num_epochs = 30
num_layers = 3
batch_size = 64
hidden_size = 256
num_steps = 20
seq_length = batch_size*num_steps
num_batches = X.shape[0] // seq_length

def detach(states):
    return [state.detach() for state in states]

model_file = 'model_30.ckpt'

# Train the model
if not os.path.isfile(model_file):
    for epoch in range(num_epochs):
        # Set initial hidden and cell states
        states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
                  torch.zeros(num_layers, batch_size, hidden_size).to(device))

        for i in range(0, X.shape[0] - seq_length - 1, seq_length):
            # Get mini-batch inputs and targets
            inputs = X[i:i + seq_length].to(device)
            targets = y[i:i + seq_length].to(device).long()
            inputs = inputs.reshape(batch_size, num_steps)
            targets = targets.reshape(batch_size, num_steps)

            # Forward pass
            states = detach(states)
            outputs, states = model(inputs, states)
            loss = criterion(outputs, targets.reshape(-1))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            # clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            step = (i + 1) // seq_length
            if step % 100 == 0:
                print('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
                      .format(epoch + 1, num_epochs, step, num_batches, loss.item(), np.exp(loss.item())))

    # Save the model checkpoints
    torch.save(model.state_dict(), model_file)



# 5. Evaluation
# test set에서 average preplexity 계산
# cross entropy
model.load_state_dict(torch.load(model_file))
print("model is loaded")

# Test the model
cnt = 0
with torch.no_grad():
    model.eval()
    test_loss = 0
    test_perplexity = 0

    states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
              torch.zeros(num_layers, batch_size, hidden_size).to(device))
    states = detach(states)
    # input = test_X[0].unsqueeze(1).to(device)

    for i in range(0, test_y.shape[0], batch_size):
        input = test_X[i:i+batch_size].to(device)
        output, states = model(input.unsqueeze(1), states)  # torch.Size([64, 1024])
        target = test_y[i:i+batch_size].to(device)      # torch.Size([64])

        loss = criterion(output, target)
        test_loss += loss
        test_perplexity += torch.exp(loss)

        if i % 1000 == 0:
            print('Test Progress[{}/{}], Loss: {:.4f}, Perplexity: {:.4f}'
                  .format(i, test_y.shape[0], loss, torch.exp(loss)))

step = len(test_X)/batch_size
print(f"test loss : {(test_loss/step).item()}")
print(f"test perplexity : {(test_perplexity/step).item()}")


# 6. 제출: ptb.vocab, model file, LSTM과 Transformer 모델 비교 리포트, train과 evaluation 코드


"""
test loss : 4.652682304382324
test perplexity : 131.58116149902344
"""




