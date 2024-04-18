"""
author : 신한이
student ID : 2024021072
date : 2024/04/17
description : Transformer decoder
"""

import sentencepiece as spm
import pandas as pd
import csv
import utils
from model import *
import os


# 1. 토큰화(sentencepiece.unigram)
if not os.path.isfile('ptb.vocab'):
    spm.SentencePieceTrainer.Train('--input=dataset/ptb.train.txt '
                               '--model_prefix=ptb '
                               '--vocab_size=1024 '   # token : 1024
                               '--model_type=unigram '
                               '--max_sentence_length=9999')

# vocab load
vocab_list = pd.read_csv('ptb.vocab', sep='\t', header=None, quoting=csv.QUOTE_NONE)
sp = spm.SentencePieceProcessor()
vocab_file = "ptb.model"
sp.load(vocab_file)


## 2. 데이터 로드(utils.py)
# [이전 토큰, 다음 토큰]
# train
train_data_path = 'dataset/ptb.train.txt'
train_X, train_y = utils.data_loader(train_data_path, vocab_file=sp)
# test
test_data_path = "dataset/ptb.test.txt"
test_X, test_y = utils.data_loader(test_data_path, vocab_file=sp)


### MAIN CODE START ###


# 3. Transformer 디코더 모델 build
device = utils.cuda_test()
model = TransformerDecoder(num_tokens=1024, dim_model=256, num_heads=4).to(device)
print(f'model : {model}')

# total parameter 계산
print('Total parameters in model: {:,}'.format(get_total_params(model)))



# 4. Train the model

## Parmaeter
num_epochs = 30
num_layers = 3
batch_size = 64
hidden_size = 256
num_steps = 20
vocab = 1024
total_loss = 0
model_file = 'transformer_30.ckpt'
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

if not os.path.isfile(model_file):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for i in range(batch_size*num_steps, train_X.shape[0]-646, batch_size*num_steps):
            inputs = train_X[i:i + batch_size*num_steps].to(device)
            targets = train_y[i:i + batch_size*num_steps].to(device).long()
            inputs = inputs.reshape(batch_size, num_steps)  # X
            targets = targets.reshape(batch_size, num_steps)   # y

            y_input = train_y[i-batch_size*num_steps:i].to(device)
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
                      .format(epoch + 1, num_epochs, i, train_X.shape[0], loss.item(), np.exp(loss.item())))

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

    for i in range(batch_size, test_y.shape[0], batch_size):
        input = test_X[i:i+batch_size].to(device)
        target = test_y[i:i+batch_size].to(device)      # torch.Size([64])

        y_input = test_y[i-batch_size:i].to(device)
        tgt_mask = model.get_tgt_mask(batch_size).to(device)

        pred = model(input, y_input, tgt_mask)

        pred = pred.permute(1, 2, 0)    # torch.Size([64, 1024, 64])
        pred = pred[:,:,0]

        loss = criterion(pred, target)

        test_loss += loss
        test_perplexity += torch.exp(loss)

        if i % 1000 == 0:
            print('Test Progress[{}/{}], Loss: {:.4f}, Perplexity: {:.4f}'
                  .format(i, test_y.shape[0], loss, torch.exp(loss)))

step = len(test_X)/batch_size
print(f"test loss : {(test_loss/step).item()}")
print(f"test perplexity : {(test_perplexity/step).item()}")

"""
Output)

device: mps
Total parameters in model: 6,840,320
test loss : 5.223451137542725
test perplexity : 200.17913818359375
"""
