import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LSTM(nn.Module):
    def __init__(self, input_size, output_size = 1, hidden_size = 10, hidden_num = 2, dropout = 0.5):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.hidden_num = hidden_num

        self.hidden = self.init_hidden()

        self.relu = nn.ReLU()
        self.lstm1 = nn.LSTM(self.input_size, self.hidden_size)
        self.lstm2 = nn.LSTM(self.hidden_size, self.hidden_size)
        self.linear = nn.Conv1d(self.hidden_size,1,1)
        self.dropout =nn.Dropout(dropout)

    def init_hidden(self):
        return (torch.zeros(1, self.hidden_size, self.hidden_size),
                torch.zeros(1, self.hidden_size, self.hidden_size))


    def forward(self, x):
        x, self.hidden = self.lstm1(x, self.hidden)
        x = self.relu(x)
        for i in range(self.hidden_num):
            x = self.dropout(x)
            x, self.hidden = self.lstm2(x, self.hidden)
            x = self.relu(x)

        pred = self.linear(x.transpose(2,1))

        return pred

def creat_input_target(x, timelag= 12, col_name = None):

    # x : dataframe
    # timelag : int
    # target_name = str, Predict column name

    input_ = []
    target = []
    L = len(input_)
    target_col = x[col_name]
    for i in range(L - timelag - 1):
        train_seq = x[i:i+timelag]
        train_target = target_col[i+timelag+1]
        input_.append(train_seq)
        target.append(train_target)

    return input_, target

def train_LSTM(model, input_, target, epochs = 100, optimizer = 'Adam', loss_fn = 'MSE', lr = 0.001):
    # optimizer : Adam, RMSProp
    # loss_fn : MSE, RMSE

    if loss_fn == 'MSE':
        loss_fn = nn.MSELoss()
    else:
        print("Warning : Wrong loss Function")
        return

    if optimizer == "Adam":
        optim_fn = optim.Adam(model.parameters(), lr=lr)
    elif optimizer == "RMSProp":
        optim_fn = optim.RMSprop(model.parameters(),lr = lr)
    else:
        print("warning : wrong optimizer")
        return


    train_loss_list = []
    for i in range(epochs):
        train_loss = 0.0

        model.train()
        model.zero_grad()
        optim_fn.zero_grad()

        for j in range(len(x)):
            seq, labels = x[j], y[j]
            seq = torch.tensor(seq[np.newaxis], dtype = torch.float32, device='cuda')
            labels = torch.tensor(labels[np.newaxis], dtype = torch.float32, device='cuda')

            model.zero_grad()
            optim_fn.zero_grad()
            model.hidden = [hidden.to('cuda') for hidden in model.init_hidden()]

            y_pred = model(seq)
            loss = loss_fn(y_pred, labels)

            loss.backward()
            optim_fn.step()

            train_loss += np.sqrt(loss.item())

        train_loss = train_loss/len(input_)
        train_loss_list.append(train_loss)

        if i % 10 == 0.0:
            print("{}번째 epoch의 train loss =".format(i),train_loss)

    return train_loss_list, model

def test_LSTM(model, x, y, device = None):
    model.eval()

    test_loss = 0.0
    x = torch.tensor(x[np.newaxis], dtype=torch.float32, device=device)
    y = torch.tensor(y[np.newaxis], dtype=torch.float32, device=device)
    with torch.no_grad():
        for seq, true_y in [x,y]:
            seq = seq.to(device)
            model.hidden = [hidden.to(device) for hidden in model.init_hidden()]
            pred_y = model(seq).cpu().detach().numpy()

            loss = (pred_y - true_y)**2
            test_loss += loss

    test_loss /= len(x)

    print("test loss :", test_loss)
    return test_loss, model















