import torch
import os
import numpy as np

def cuda_test():
    if os.name == 'nt':
        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    elif os.name == 'posix':
        device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
    else:
        print("Unsupported operating")

    print(f"device: {device}")

    return device

def detach(states):
    return [state.detach() for state in states]

def data_loader(file_path, vocab_file):
    # file read
    lines = []
    f = open(file_path, 'r')
    while True:
        line = f.readline()
        if not line:
            break
        lines.append(line)
    f.close()

    encoded_train = []
    for line in lines:
        encoded_train.append(vocab_file.encode_as_ids(line))

    sequences = []
    for i in range(0, len(encoded_train)):
        for j in range(1, len(encoded_train[i])):
            if len(encoded_train[i]) > 1:
                sequence = encoded_train[i][j - 1:j + 1]
                sequences.append(sequence)
    sequences = np.array(sequences)

    X, y = sequences[:, 0], sequences[:, 1]

    return torch.from_numpy(X), torch.from_numpy(y)

