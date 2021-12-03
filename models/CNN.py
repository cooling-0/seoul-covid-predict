import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class CNN(nn.Module):
    def __init__(self, input_size, output_size = 1):
        super(CNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.conv1 = nn.Conv1d(1, 1, kernel_size = (1, self.input_size), stride = 1, padding = 1)
        self.conv2 = nn.Covn1d(1, 1, kernel_size = (2,3), stride = 1, padding = 1)
        self.relu = nn.ReLU()
        # self.batchnorm = nn.BatchNorm1d()
        # self.pool1 = nn.AvgPool1d(2, 2)
        self.fc1 = nn.Linear(51, self.output_size)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.pool1(x)
        # x = self.batchnorm(x)

        x = self.relu(x)
        x = self.conv2(x)
        x = self.fc1(x)
        return x

'''
x = np.random.rand(1,14,26)
x = torch.tensor(x[np.newaxis], dtype=torch.float32, device='cpu')
print(x.size())

conv1 = nn.Conv1d(1, 1, kernel_size = (1,26), stride = 1, padding = 1)
conv2 = nn.Conv1d(1, 1, kernel_size = (2,3), stride = 1, padding = 1)
pool1 = nn.AvgPool1d(2, 2)
relu = nn.ReLU()
linear = nn.Linear(51,1)

x = conv1(x)
x = relu(x)
x.size()
x = conv2(x)
x.size()
x = torch.flatten(x)
x = linear(x)
x
'''

def train(model, data, target, epochs = 100 ,lr = 0.001, l2 = 0.00001):
    # optimizer  = adam , Loss_fn = MSELoss

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay = l2)
    criterion = nn.MSELoss()

    train_list = []
    for epoch in range(epochs):
        train_loss = 0.0
        model.train()
        optimizer.zero_grad()
        model.zero_grad()

        for i in range(len(data)):
              # get the inputs
              input, label = data[i], target[i]
              input = inputs.cuda()
              label = label.cuda()
              output = model(input)

              loss = criterion(output, label)
              loss.backward()
              optimizer.step()

              train_loss += loss.item()
        train_loss /= len(data)
        train_list.append(train_loss)
        print('{}번째 loss :'.format(epoch),train_loss)


    return model, train_list