"""
author : 신한이
student ID : 2024021072
date : 2024/04/17
description : LSTM decoder
"""

import sentencepiece as spm
import pandas as pd
import csv
import utils
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
sp = spm.SentencePieceProcessor()
vocab_file = "ptb.model"
sp.load(vocab_file)


# 2. 데이터 로드(utils.py)
# [이전 토큰, 다음 토큰]
# train
train_data_path = 'dataset/ptb.train.txt'
train_X, train_y = utils.data_loader(train_data_path, vocab_file=sp)
# test
test_data_path = "dataset/ptb.test.txt"
test_X, test_y = utils.data_loader(test_data_path, vocab_file=sp)


### MAIN CODE START ###


# 3. LSTM 디코더 모델 build
device = utils.cuda_test()
model = DecoderRNN(vocab_size=1024, embed_size=256, hidden_size=256, num_layers=3).to(device)
print(f'model : {model}')

# total parameter 계산
print('Total parameters in model: {:,}'.format(get_total_params(model)))



# 4. Train the model
criterion = F.cross_entropy
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
model.train()

# hyper parameters
num_epochs = 30
num_layers = 3
batch_size = 64
hidden_size = 256
num_steps = 20
seq_length = batch_size*num_steps
num_batches = train_X.shape[0] // seq_length


def detach(states):
    return [state.detach() for state in states]

model_file = 'model_30.ckpt'

# train the model
if not os.path.isfile(model_file):
    for epoch in range(num_epochs):
        # Set initial hidden and cell states
        states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
                  torch.zeros(num_layers, batch_size, hidden_size).to(device))

        for i in range(0, train_X.shape[0] - seq_length - 1, seq_length):
            # Get mini-batch inputs and targets
            inputs = train_X[i:i + seq_length].to(device)
            targets = train_y[i:i + seq_length].to(device).long()
            inputs = inputs.reshape(batch_size, num_steps)
            targets = targets.reshape(batch_size, num_steps)

            # Forward pass
            states = detach(states)
            outputs, states = model(inputs, states)
            loss = criterion(outputs, targets.reshape(-1))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step = (i + 1) // seq_length
            if step % 1000 == 0:
                print('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
                      .format(epoch + 1, num_epochs, step, num_batches, loss.item(), np.exp(loss.item())))

    # Save the model checkpoints
    torch.save(model.state_dict(), model_file)



# 5. Evaluation
model.load_state_dict(torch.load(model_file))
print("model is loaded")

# Test the model
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
Output)

device: mps
Total parameters in model: 2,104,320
test loss : 4.652682304382324
test perplexity : 131.58116149902344
"""




