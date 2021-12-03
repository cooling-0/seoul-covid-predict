
import pandas as pd
import numpy as np
def creat_input_target(x,y,dic, timelag= 14, col_name = None):

    # x : dataframe
    # timelag : int
    # target_name = str, Predict column name

    input_ = []
    target = []
    L = len(x)
    x = x[dic[col_name]]
    target_col = y[col_name]
    for i in range(L - timelag - 1):
        train_seq = x[i:i+timelag].values
        train_target = target_col[i+timelag+1]
        input_.append(train_seq)
        target.append(train_target)

    return np.array(input_), np.array(target)

import random
import numpy as np
import torch


class MV_LSTM(torch.nn.Module):
    def __init__(self, n_features, seq_length):
        super(MV_LSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = 20
        self.n_layers = 1

        self.l_lstm = torch.nn.LSTM(input_size=n_features,
                                    hidden_size=self.n_hidden,
                                    num_layers=self.n_layers,
                                    batch_first=True)

        self.l_linear = torch.nn.Linear(self.n_hidden * self.seq_len, 1)

    def init_hidden(self, batch_size):
        hidden_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        cell_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        self.hidden = (hidden_state, cell_state)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        lstm_out, self.hidden = self.l_lstm(x, self.hidden)

        x = lstm_out.contiguous().view(batch_size, -1)
        return self.l_linear(x)


def train_mv_lstm(model, criterion, optimizer, epochs, batch_size):
    pred_list_list = []
    model.train()
    loss_list = []
    for t in range(epochs):
        pred_list = []
        loss_list_ep = []
        for b in range(0, len(x_train), batch_size):
            inpt = x_train[b:b + batch_size, :, :]
            target = y_train[b:b + batch_size]

            x_batch = torch.tensor(inpt, dtype=torch.float32)
            y_batch = torch.tensor(target, dtype=torch.float32)

            mv_net.init_hidden(x_batch.size(0))
            #    lstm_out, _ = mv_net.l_lstm(x_batch,nnet.hidden)
            #    lstm_out.contiguous().view(x_batch.size(0),-1)
            output = model(x_batch)
            pred_list.append(output)
            loss = criterion(output.view(-1), y_batch)
            loss_list_ep.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        loss_list.append(sum(loss_list_ep) / len(x_train))
        pred_list_list.append(pred_list)
        print('step : ', t, 'loss : ', sum(loss_list_ep) / len(x_train))

    return model, loss_list, pred_list_list

"""
# example code
# this is seoul borough's nearest
dic_ = {'강서구' : ['강서구','양천구'],
        '양천구' : ['양서구','강서구', '영등포구', '구로구'],
        '구로구' : ['구로구','양천구', '영등포구', '금천구'],
        '영등포구' : ['영등포구','양천구', '구로구', '동작구'],
        '금천구' : ['금천구','구로구', '관악구'],
        '동작구' : ['동작구','영등포구', '관악구', '서초구'],
        '관악구' : ['관악구','금천구', '동작구', '서초구'],
        '서초구' : ['서초구','동작구', '관악구', '강남구'],
        '강남구' : ['강남구','서초구', '송파구'],
        '송파구' : ['송파구','강남구', '강동구'],
        '강동구' : ['강동구','송파구'],
        '은평구' : ['은평구','종로구', '서대문구', '마포구'],
        '서대문구' : ['서대문구','종로구', '중구', '은평구', '마포구'],
        '마포구' : ['마포구','은평구', '서대문구', '용산구'],
        '종로구' : ['종로구','은평구', '서대문구', '중구', '성북구', '동대문구'],
        '중구' : ['중구','종로구', '서대문구', '용산구', '성동구'],
        '용산구' : ['용상구','마포구', '중구', '성동구'],
        '성동구' : ['성동구','용산구', '중구', '동대문구', '광진구'],
        '광진구' : ['광진구','성동구', '동대문구', '중랑구'],
        '동대문구'  : ['동대문구','성동구', '광진구', '중랑구', '성북구', '종로구'],
        '중랑구' : ['중랑구','광진구', '동대문구', '성북구', '노원구'],
        '노원구' : ['노원구','중랑구', '성북구', '강북구', '도봉구'],
        '도봉구' : ['도봉구','강북구', '노원구'],
        '강북구' :['강북구','도봉구', '노원구', '성북구'],
        '성북구' : ['성북구','강북구', '노원구', '중랑구', '동대문구', '종로구']}

# example code

df = pd.read_csv('/content/seoul_count.csv')
df1 = df[:500]
df2 = df[501:].reset_index(drop = True)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
sc.fit(df1)
df1_ = sc.transform(df1)
df1_ = pd.DataFrame(df1_,columns = df.columns)
df2_ = sc.transform(df2)
df2_ = pd.DataFrame(df2_,columns = df.columns)
x_train, y_train = creat_input_target(df1_,df1,dic_,14,'도봉구')
x_test, y_test = creat_input_target(df2_,df2,dic_,14,'도봉구')
   
n_features =  x_train.shape[-1]# this is number of parallel inputsaa
n_timesteps = 14 # this is number of timesteps

# create NN
mv_net = MV_LSTM(n_features,n_timesteps)
criterion = torch.nn.MSELoss() # reduction='sum' created huge loss value
optimizer = torch.optim.Adam(mv_net.parameters(), lr=1e-3,weight_decay=1e-2)

train_episodes = 300
batch_size = 1
"""