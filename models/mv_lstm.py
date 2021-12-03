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

'''
# example training code

n_features = 26 # this is number of parallel inputs
n_timesteps = 14 # this is number of timesteps

# create NN
mv_net = MV_LSTM(n_features,n_timesteps)
criterion = torch.nn.MSELoss() # reduction='sum' created huge loss value
optimizer = torch.optim.Adam(mv_net.parameters(), lr=1e-2,weight_decay=1e-5)

train_episodes = 100
batch_size = 1

pred_list_list = []
mv_net.train()
loss_list = []
for t in range(train_episodes):
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
        output = mv_net(x_batch)
        pred_list.append(output)
        loss = criterion(output.view(-1), y_batch)
        loss_list_ep.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    loss_list.append(sum(loss_list_ep) / len(x_train))
    pred_list_list.append(pred_list)
    print('step : ', t, 'loss : ', sum(loss_list_ep) / len(x_train))
'''