import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.normalized_shape = tuple(normalized_shape)
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, input):
        mean = input.mean(dim=(1, 2), keepdim=True)
        variance = input.var(dim=(1, 2), unbiased=False, keepdim=True)
        input = (input - mean) / torch.sqrt(variance + self.eps)
        if self.elementwise_affine:
            input = input * self.weight + self.bias
        return input


class GLU(nn.Module):
    def __init__(self, features, dropout=0.1):
        super(GLU, self).__init__()
        self.conv1 = nn.Conv2d(features, features, (1, 1))
        self.conv2 = nn.Conv2d(features, features, (1, 1))
        self.conv3 = nn.Conv2d(features, features, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        out = x1 * torch.sigmoid(x2)
        out = self.dropout(out)
        out = self.conv3(out)
        return out


class Conv(nn.Module):
    def __init__(self, features, dropout=0.1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(features, features, (1, 1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class TemporalEmbedding(nn.Module):
    def __init__(self, time, features):
        super(TemporalEmbedding, self).__init__()

        self.time = time
        # temporal embeddings
        self.time_day = nn.Parameter(torch.empty(time, features))
        nn.init.xavier_uniform_(self.time_day)

        self.time_week = nn.Parameter(torch.empty(7, features))
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, x):
        day_emb = x[..., 1]  
        time_day = self.time_day[
            (day_emb[:, -1, :] * self.time).type(torch.LongTensor)
        ]  
        time_day = time_day.transpose(1, 2).unsqueeze(-1)

        week_emb = x[..., 2]  
        time_week = self.time_week[
            (week_emb[:, -1, :]).type(torch.LongTensor)
        ]  
        time_week = time_week.transpose(1, 2).unsqueeze(-1)

        tem_emb = time_day + time_week
        return tem_emb



class Encoder(nn.Module):
    def __init__(self, d_model, head, num_nodes, seq_length=1, dropout=0.1,lamda=0.1):
        "Take in model size and number of heads."
        super(Encoder, self).__init__()
        assert d_model % head == 0
        self.d_k = d_model // head  # We assume d_v always equals d_k
        self.head = head
        self.num_nodes = num_nodes
        self.seq_length = seq_length
        self.d_model = d_model
        self.attention = MultiHeadDifferentialAttention(d_model, head, lamda, num_nodes)
        self.LayerNorm = LayerNorm(
            [d_model, num_nodes, seq_length], elementwise_affine=False
        )
        self.dropout1 = nn.Dropout(p=dropout)
        self.glu = GLU(d_model)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, input, adj_list=None):
        # 64 64 170 12
        x, weight, bias = self.attention(input)
        x = x + input
        x = self.LayerNorm(x)
        x = self.dropout1(x)
        x = self.glu(x) + x
        x = x * weight + bias + x
        x = self.LayerNorm(x)
        x = self.dropout2(x)
        return x


class DMT(nn.Module):
    def __init__(
        self,
        input_dim=3,
        channels=64,
        num_nodes=170,
        input_len=12,
        output_len=12,
        dropout=0.1,
        lamda=0.1,
        layer=1,
        head=1,
        # time=288,
    ):
        super().__init__()

        # attributes
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.input_dim = input_dim
        self.output_len = output_len
        self.head = head
        
        if num_nodes == 716:
            time = 96
        else:
            time =288

        self.Temb = TemporalEmbedding(time, channels)

        self.start_conv = nn.Conv2d(self.input_dim, channels, kernel_size=(1, 1))
        self.conv = nn.Conv2d(channels, channels, kernel_size=(1, 12))

        self.network_channel = channels * 2

        self.SpatialBlock = nn.ModuleList(
            [
                Encoder(
                    d_model=self.network_channel,
                    head=self.head,
                    num_nodes=num_nodes,
                    seq_length=1,
                    dropout=dropout,
                    lamda=lamda
                )
                for _ in range(layer)
            ]
        )
        self.fc_st = nn.ModuleList(
            [
                nn.Conv2d(
                    self.network_channel, self.network_channel, kernel_size=(1, 1)
                )
                for _ in range(layer)
            ]
        )

        self.regression_layer = nn.Conv2d(
            self.network_channel, self.output_len, kernel_size=(1, 1)
        )

    
    def forward(self, history_data: torch.Tensor) -> torch.Tensor:
        # b,t,n,d
        input_data = history_data
        input_data = self.start_conv(input_data)

        input_data = self.conv(input_data) 
        history_data = history_data.permute(0, 3, 2, 1)
        tem_emb = self.Temb(history_data)

        data_st = torch.cat([input_data] + [tem_emb], dim=1)

        for SpatialBlock, fc_st in zip(self.SpatialBlock, self.fc_st):
            data_st = SpatialBlock(data_st) + fc_st(data_st)
        prediction = self.regression_layer(data_st)

        return prediction
    
    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])
