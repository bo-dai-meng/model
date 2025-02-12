import math, random
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l
from torch.utils import data
from torch.utils.data import DataLoader

# =====scaled dot-product attention=======
def scaled_dot_product(q, k, v, mask = None):
    dim_k = q.size()[-1]
    qk_dot_product = torch.matmul(q, k.transpose(-2, -1))
    dot_product = qk_dot_product / math.sqrt(dim_k)
    if mask is not None:
        dot_product = dot_product.masked_fill(mask==0, -9e15)
    attention = nn.functional.softmax(dot_product, dim=-1)
    values = torch.matmul(attention, v)
    return values

# ==== Attention=========================
class Attention(nn.Module):
    def __init__(self, dim_input: int, dim_q: int, dim_k: int):
        super().__init__()
        self.linear_q = nn.Linear(dim_input, dim_q)
        self.linear_k = nn.Linear(dim_input, dim_k)
        self.linear_v = nn.Linear(dim_input, dim_k)

    def forward(self, query, key, value, mask=None):
        return scaled_dot_product(self.linear_q(query), self.linear_k(key), self.linear_v(value), mask)

# ===== num_heads=2 =====================
class MutiHeadAtt(nn.Module):
    def __init__(self, num_heads, dim_input, dim_q, dim_k):
        super().__init__()
        self.heads1 = Attention(dim_input, dim_q, dim_k)
        self.heads2 = Attention(dim_input, dim_q, dim_k)
        self.linear = nn.Linear(num_heads * dim_k, dim_input)

    def forward(self, query, key, value, mask=None):
        head1 = self.heads1(query, key, value, mask)
        head2 = self.heads2(query, key, value, mask)
        output = torch.cat([head1, head2], dim=-1)
        return self.linear(output)

# =========Feedforward================
def feed_forward(input_dim, intermediate_dim):

    return nn.Sequential(
        nn.Linear(input_dim, intermediate_dim),
        nn.ReLU(),
        nn.Linear(intermediate_dim, input_dim)
    )

# =======Add&Norm=====================
class AddNorm(nn.Module):
    def __init__(self, sublayer, dim, dropout_rate=0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, tensors, *args, **kwargs):
        return self.norm(tensors + self.dropout(self.sublayer(tensors, *args, **kwargs)))

# =====PositionalEncoding==============
class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

# =====Encoder block==================
class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, feedforward_dim, dropout_rate):
        super().__init__()
        query_dim = key_dim = int(max(input_dim / num_heads, 1))
        self.multi_head_attention = AddNorm(
            MutiHeadAtt(num_heads, input_dim, query_dim, key_dim),
            dim=input_dim,
            dropout_rate=dropout_rate
        )
        self.feedforward_network = AddNorm(
            feed_forward(input_dim, feedforward_dim),
            dim=input_dim,
            dropout_rate=dropout_rate
        )
    def forward(self, src, mask=None):
        output = self.multi_head_attention(src, src, src, mask)
        final_output = self.feedforward_network(output)
        return final_output

# =====Encoder======================
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, input_dim, num_heads, feedforward_dim, dropout_rate):
        super().__init__()
        self.layer= nn.ModuleList(
            [TransformerEncoderLayer(input_dim, num_heads, feedforward_dim, dropout_rate) for i in range(num_layers)]
        )
        self.pos_lay = PositionalEncoding(input_dim)
    def forward(self, src, mask=None):
        seq_len, input_dim = src.size(1), src.size(2)
        input = (src + self.pos_lay(src))
        for layer in self.layer:
            enc_output = layer(input, mask)
        return enc_output

# ======Decoder block=================
class TransformerDecoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, feedforward_dim, dropout_rate):
        super().__init__()
        query_dim = key_dim = int(max(input_dim / num_heads, 1))
        self.self_attention = AddNorm(
            MutiHeadAtt(num_heads, input_dim, query_dim, key_dim),
            dim=input_dim,
            dropout_rate=dropout_rate
        )

        self.cross_attention = AddNorm(
            MutiHeadAtt(num_heads, input_dim, query_dim, key_dim),
            dim=input_dim,
            dropout_rate=dropout_rate
        )
        self.feed_forward = AddNorm(
            feed_forward(input_dim, feedforward_dim),
            dim=input_dim,
            dropout_rate=dropout_rate
        )
    def forward(self, target, enc_output, mask):
        output = self.self_attention(target, target, target, mask)
        output = self.cross_attention(output, enc_output, enc_output)
        return self.feed_forward(output)

# ======Decoder======================
class TransformerDecoder(nn.Module):
    def __init__(self, num_layer, input_dim, num_heads, feedforward_dim, dropout_rate):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(input_dim, num_heads, feedforward_dim, dropout_rate) for i in range(num_layer)]
        )
        self.FC = nn.Linear(input_dim, 1)
        self.Pos = PositionalEncoding(input_dim)
    def forward(self, target, enc_output, mask):
        input = self.Pos(target) + target
        for layer in self.layers:
            FC_input = layer(input, enc_output, mask)
        # output_dim = 1
        return self.FC(FC_input)

# ======Transformer================
class Transformer_Encoder_Decoder(nn.Module):
    def __init__(self, num_layer: int, input_dim: int, num_heads: int, feedforward_dim: int, dropout_rate):
        super().__init__()
        self.encoder = TransformerEncoder(
            num_layers=num_layer,
            input_dim=input_dim,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            dropout_rate=dropout_rate
        )
        self.decoder = TransformerDecoder(
            num_layer=num_layer,
            input_dim=input_dim,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            dropout_rate=dropout_rate
        )
    def forward(self, src_input, target, mask):
        enc_output = self.encoder(src_input)

        return self.decoder(target, enc_output, mask)

# =====sequence mask===========
def get_masks(timestep):
    return torch.tril(torch.ones(timestep, timestep))

# ====scheduled sampling==========
def flip_from_probability(p):
    return True if random.random() < p else False

time_step = 10
hidden_sizes = 10
num_layers = 8

# =====xavier initialization======
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        nn.init.xavier_uniform_(m.weight)

def load_data(path):
    df = pd.read_excel(path).values
    dataset = torch.from_numpy(df)
    timestep = 20
    sequence_feature_list = []
    for index in range(len(dataset) - timestep - 1):
        sequence_feature_list.append(dataset[index: index + timestep])
    sequence_feature = torch.stack(sequence_feature_list)

    sequence_label_list = []
    for index in range(len(dataset) - timestep - 1):
        sequence_label_list.append(dataset[index + 1: index + timestep + 1, 3])
    sequence_label = torch.stack(sequence_label_list)
    return sequence_feature, sequence_label

lr = 0.005
batch_sizes = 1707
epoch = 3000

# ======load data(train dataset)==========================
sequence_feature, sequence_label= load_data("D:\\Al-ion\\transformer\\1-train.xlsx")
dataset = data.TensorDataset(sequence_feature, sequence_label)
dataloader = DataLoader(dataset, batch_size= batch_sizes, shuffle=False, drop_last=True)

emodel = Transformer_Encoder_Decoder(
    num_layer=num_layers, input_dim=4, num_heads=2, feedforward_dim=10, dropout_rate=0.1).to(device=torch.device('cuda'))
emodel.apply(init_weights)
loss = nn.MSELoss().to(device=torch.device('cuda'))
optimizer = torch.optim.Adam(emodel.parameters(), lr, weight_decay=0.01)
label_list = []

# ====train=================================
emodel.train()
train_loss_list = []
for i in range(epoch):
    count = 0
    for datas in dataloader:
        count = count + 1
        optimizer.zero_grad()
        feature, label = datas
        feature = feature.float().cuda()
        label = label.float().cuda()
        print(f"\nfeature_shape：{feature.shape}, label_shape：{label.shape}")
        output = emodel(feature[:, 0:10, :], feature[:, 10:20, :], get_masks(time_step).cuda())
        print(f"\nprediction:{output}，output_shape：{output.shape}")
        train_loss = loss(output, label.reshape(batch_sizes, 20, 1)[:, 10:20, :])
        print(f"\nloss:{train_loss}")
        train_loss_list.append(train_loss.item())
        train_loss.backward()
        optimizer.step()

# ======load data(test dataset)==========================
sequence_feature1, sequence_label1= load_data("D:\\Al-ion\\transformer\\1-test.xlsx")
dataset1 = data.TensorDataset(sequence_feature1, sequence_label1)

# ====test===================================
emodel.eval()
with torch.no_grad():
    output_list = []
    test_dataloader = DataLoader(dataset1, batch_size=1, shuffle=False, drop_last=True)
    for datas in test_dataloader:
        feature, label = datas
        print(feature.shape, label.shape)
        label = label[0, -1]
        label_list.append(label.item())
        feature = feature.float().cuda()
        label = label.float().cuda()
        output = emodel(feature[:, 0:10, :], feature[:, 10:20, :], None)
        output = output[0, -1].cpu()
        print(output, output.shape)
        output_list.append(output.item())

# =====plot===============================
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
    ax2.set_xlabel("Cycle", fontsize=25)
    ax2.set_ylabel("Capacity(mAh/g)", fontsize=25)
    print(len(output_list))
    plt_data = plt.plot(x2, label_list, color="dodgerblue", label="data", linewidth=2, linestyle="-")
    plt.tick_params(labelsize=15)
    plt.legend(prop={"size": 20})
    plt.show()

plot(12, 10)

