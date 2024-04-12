
import numpy as np
import random
import os, errno
import sys

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


# RNN based language model
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h):
        # Embed word ids to vectors
        x = self.embed(x)

        # Forward propagate LSTM
        out, (h, c) = self.lstm(x, h)

        # Reshape output to (batch_size*sequence_length, hidden_size)
        out = out.reshape(out.size(0) * out.size(1), out.size(2))

        # Decode hidden states of all time steps
        out = self.linear(out)
        return out, (h, c)




def keras_model():
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Embedding
    from keras.callbacks import ModelCheckpoint
    from keras.models import load_model
    from collections import Counter

    model = Sequential()
    model.add(Embedding(5000, 256, input_length=1))
    model.add(LSTM(256))
    model.add(Dense(256, activation='softmax'))
    print(model.summary())

    #  모델 컴파일 (추가 설명 예정)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 학습이 오래 걸릴 경우를 대비해서 각 epoch 수행 후 결과를 파일에 저장한다.
    checkpoint_path = "models/term.weights.h5"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                  save_weights_only=True,
                                  verbose=1)

    # 모델 훈련
    model.fit(X, y, epochs=10, verbose=2, callbacks=[cp_callback])
    model.save("models/Model")
