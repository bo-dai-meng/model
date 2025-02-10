import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torch.utils import data

class LSTMLayer(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.forget_gate = nn.Linear(input_dim + hidden_size, hidden_size)
        self.input_gate = nn.Linear(input_dim + hidden_size, hidden_size)
        self.cell_state = nn.Linear(input_dim + hidden_size, hidden_size)
        self.output_gate = nn.Linear(input_dim + hidden_size, hidden_size)

    def forward(self, input, state = None):
        batch_size = input.size(0)
        if state is None:
            state = self.init_state(batch_size, input.device)
        hidden_state, cell_state = state
        union_input = torch.concat((input, hidden_state), dim=-1)
        ingate = torch.nn.functional.sigmoid(self.input_gate(union_input))
        forgetgate = torch.nn.functional.sigmoid(self.forget_gate(union_input))
        new_cell_state = torch.nn.functional.tanh(self.cell_state(union_input))
        final_cell_state = (cell_state * forgetgate) + (ingate * new_cell_state)
        outgate = torch.nn.functional.sigmoid(self.output_gate(union_input))
        final_hidden_state = outgate * torch.nn.functional.tanh(final_cell_state)
        return final_hidden_state, final_cell_state

    def init_state(self, batch_size, device):
        hidden_state = torch.zeros((batch_size, self.hidden_size), device=device)
        cell_state = torch.zeros((batch_size, self.hidden_size), device=device)
        return hidden_state, cell_state

def test_LSTM_Model():
    lstm_layer = LSTM_Model(3, 4)
    lstm_layer = lstm_layer.to(device=torch.device('cuda'))
    x = torch.randn(5, 10, 3, device=torch.device('cuda'))
    a = lstm_layer(x)
    return a.shape

class LSTM_Model(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.lstmlayer = LSTMLayer(input_dim, hidden_size)
        self.FC = nn.Linear(hidden_size, 1)

    def forward(self, input, state = None):
        batch = input.size(0)
        timestep = input.size(1)
        feature = input.size(2)
        hidden_state_list = []
        for i in range(timestep):
            state = self.lstmlayer(input[:, i, :], state)
            hidden_state_list.append(state[0])
            hidden = torch.stack(hidden_state_list, dim=1)
        return hidden

class MultiLayer_Model(nn.Module):
    def __init__(self, timestep, input_dim, hidden_size):
        super().__init__()
        self.Conv1d1 = nn.Conv1d(in_channels=timestep, out_channels=timestep, kernel_size=1, stride=1)
        self.Conv1d2 = nn.Conv1d(in_channels=timestep, out_channels=timestep, kernel_size=1, stride=1)
        self.batchnorm1 = nn.BatchNorm1d(timestep)
        self.batchnorm2 = nn.BatchNorm1d(timestep)
        self.relu = nn.ReLU()
        self.lstm1 = LSTM_Model(input_dim, hidden_size)
        self.laynorm1 = nn.LayerNorm(normalized_shape=hidden_size)
        self.lstm2 = LSTM_Model(input_dim=input_dim, hidden_size=hidden_size)
        self.laynorm2 =nn.LayerNorm(hidden_size)
        self.lstm3 = LSTM_Model(input_dim=input_dim, hidden_size=hidden_size)
        self.laynorm3 = nn.LayerNorm(hidden_size)
        self.lstm4 = LSTM_Model(input_dim=input_dim, hidden_size=hidden_size)
        self.laynorm4 = nn.LayerNorm(hidden_size)
        self.lstm5 = LSTM_Model(input_dim, hidden_size)
        self.dp = nn.Dropout(0.1)
        self.FC = nn.Linear(hidden_size, 1)

    def forward(self, input, state = None):
        output = self.dp(self.lstm1(input))
        output = self.dp(self.lstm2(output))
        output = self.dp(self.lstm3(output))
        output = self.dp(self.lstm4(output))
        output = self.lstm5(output)
        final_output = self.FC(output[:, -1, :])
        return final_output

def test_MutiLayer_Model():
    model = MultiLayer_Model(timestep=10, input_dim=3, hidden_size=3)
    x = torch.randn(5, 10, 3)
    a = model(x)
    print(a.shape)

hidden_sizes = 2
num_layers = 4
time_step = 10

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        nn.init.xavier_uniform_(m.weight)

def load_data(path):
    df = pd.read_excel(path).values
    dataset = torch.from_numpy(df)
    timestep = time_step
    sequence_feature_list = []
    for index in range(len(dataset) - timestep):
        sequence_feature_list.append(dataset[index: index + timestep])
    sequence_feature = torch.stack(sequence_feature_list)

    sequence_label_list = []
    for index in range(len(dataset) - timestep):
        sequence_label_list.append(dataset[index + timestep][1])
    sequence_label = torch.stack(sequence_label_list)
    return sequence_feature, sequence_label

lr = 0.1
batch_sizes = 1714
epoch = 200

sequence_feature, sequence_label= load_data()
dataset = data.TensorDataset(sequence_feature, sequence_label)
dataloader = DataLoader(dataset, batch_size= batch_sizes, shuffle=False, drop_last=True)

emodel = MultiLayer_Model(timestep=time_step, input_dim=2, hidden_size=hidden_sizes).to(device=torch.device('cuda'))
emodel.apply(init_weights)
loss = nn.MSELoss().to(device=torch.device('cuda'))
optimizer = torch.optim.Adam(emodel.parameters(), lr, weight_decay=0.01)
label_list = []

emodel.train()
train_loss_list = []
for i in range(epoch):
    for datas in dataloader:
        optimizer.zero_grad()
        feature, label = datas
        feature = feature.float().cuda()
        label = label.float().cuda()
        output = emodel(feature)
        print(f"prediction:{output}")
        label = label.reshape(batch_sizes, -1)
        train_loss = loss(output, label)
        print(train_loss)
        train_loss_list.append(train_loss.item())
        train_loss.backward()
        optimizer.step()

sequence_feature1, sequence_label1= load_data()
dataset1 = data.TensorDataset(sequence_feature1, sequence_label1)

emodel.eval()
with torch.no_grad():
    output_list = []
    test_dataloader = DataLoader(dataset1, batch_size=1, shuffle=False, drop_last=True)
    for datas in test_dataloader:
        feature, label = datas
        label_list.append(label.item())
        feature = feature.float()
        label = label.float()
        feature = feature.cuda()
        label = label.cuda()
        output = emodel(feature)
        output = output.cpu()
        print(output, output.shape)
        output_list.append(output.item())

def plot(x_label :int, y_label: int):
    plt.figure(figsize=(x_label, y_label))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.grid(visible=True, which="major", linestyle="-", linewidth=1.5)
    plt.grid(visible=True, which="minor", linestyle="--", alpha=0.5, linewidth=1.5)
    plt.minorticks_on()
    x1 = torch.arange(0, len(train_loss_list))
    plt_loss = plt.plot(x1, train_loss_list, color="green", label="train loss", linewidth=1, linestyle="-")
    ax1 = plt.gca()
    ax1.set_title("train_loss", fontsize=20)
    ax1.set_xlabel("step", fontsize=20)
    ax1.set_ylabel("loss", fontsize=20)
    plt.tick_params(labelsize=15)

    plt.figure(figsize=(x_label, y_label))
    plt.grid(visible=True, which="major", linestyle="-", linewidth=1.5)
    plt.grid(visible=True, which="minor", linestyle="--", alpha=0.5, linewidth=1.5)
    plt.minorticks_on()
    x3 = torch.arange(len(output_list)) + 1
    x2 = torch.arange(len(label_list)) + 1
    plt_pred = plt.plot(x3, output_list, color="red", label="prediction", linewidth=2, linestyle="--")
    ax2 = plt.gca()
    ax2.set_title("prediction", fontsize=22)
    ax2.set_xlabel("Cycle", fontsize=25)
    ax2.set_ylabel("Capacity(mAh/g)", fontsize=25)
    print(len(output_list))
    plt_data = plt.plot(x2, label_list, color="dodgerblue", label="data", linewidth=2, linestyle="-")
    plt.tick_params(labelsize=15)
    plt.legend(prop={"size": 20})
    plt.show()

