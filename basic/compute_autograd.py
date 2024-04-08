
import torch
import torch.nn as nn

# Tensor 무작위 생성(0-1사이 값) (10, 4) and (10, 2).
x = torch.randn(20, 4)
y = torch.randn(20, 2)

# FCL(Fully Connected Layer)
linear = nn.Linear(4, 2)  # (In, Out) 차원
print('w: ', linear.weight)
print('b: ', linear.bias)

# loss는 MSE, Optimizer는 SGD 사용(lr=0.01)
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=learning_rate)

# Forward pass. 순전파 1번
pred = linear(x)

# Compute loss. 로스 계산
loss = criterion(pred, y)
print('loss: ', loss.item())

# Backward pass. 역전파 1번
loss.backward()

# Gradients 계산
print('dL/dw: ', linear.weight.grad)
print('dL/db: ', linear.bias.grad)

# gradient descent 계산하기(1)
optimizer.step()

# gradient descent 계산하기(2)
# linear.weight.data.sub_(learning_rate * linear.weight.grad.data)
# linear.bias.data.sub_(learning_rate * linear.bias.grad.data)

# loss 확인하기
pred = linear(x)
loss = criterion(pred, y)
print('1-step gradient descent 후의 손실(loss): ', loss.item())


# output
"""
w:  Parameter containing:
tensor([[ 0.1826,  0.2855, -0.1616,  0.2373],
        [-0.3570,  0.4438, -0.2837,  0.0687]], requires_grad=True)
b:  Parameter containing:
tensor([-0.2939, -0.4498], requires_grad=True)
loss:  2.560213565826416 (역전파 전)

dL/dw:  tensor([[ 0.3788,  0.4145, -0.5213,  0.3678],
        [-1.4561, -0.1027, -0.3621,  0.5440]])
dL/db:  tensor([-0.3812, -1.1880])
loss after 1 step optimization:  2.5122616291046143 (역전파 후 : loss가 줄어들음)

"""
