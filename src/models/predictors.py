import torch
import torch.nn as nn
import torch.nn.functional as funct
from torch.autograd import Variable

import constants.main_constants as const
import constants.sql_constants as sql_const


def init_hidden(num_layers, batch_size, rnn_size, gpu=False):
    hidden_state = Variable(torch.zeros(num_layers, batch_size, rnn_size))
    if gpu:
        hidden_state = hidden_state.cuda()

    return hidden_state


class AggregatePredictor(nn.Module):
    def __init__(self, embedding_layer, args):
        super().__init__()
        self.args = args
        self.embedding_layer = embedding_layer

        self.dropout = nn.Dropout(const.AGG_CNN_DROPOUT)
        self.cnn_layer = nn.Conv2d(in_channels=1, out_channels=const.AGG_CNN_NUM_FILTERS,
                                   kernel_size=const.AGG_CNN_KERNEL_SIZE, stride=const.AGG_CNN_STRIDE,
                                   padding=(int(const.AGG_CNN_KERNEL_SIZE[0] / 2), 0))
        self.rnn_layer = nn.GRU(args.emb_size, hidden_size=const.AGG_RNN_SIZE, num_layers=const.AGG_RNN_LAYERS,
                                dropout=const.AGG_RNN_DROPOUT)
        self.rnn_hidden_state = init_hidden(const.AGG_RNN_LAYERS, args.batch_size, const.AGG_RNN_SIZE, args.gpu)

        self.dense_layer_1 = nn.Linear(const.AGG_RNN_SIZE, int(const.AGG_RNN_SIZE / 2))
        self.dense_layer_2 = nn.Linear(int(const.AGG_RNN_SIZE / 2), len(sql_const.AGG_OPERATORS))

    def reset_hidden_state(self):
        self.rnn_hidden_state = init_hidden(const.AGG_RNN_LAYERS, self.args.batch_size, const.AGG_RNN_SIZE,
                                            self.args.gpu)

    def forward(self, input):
        input_embeddings = self.embedding_layer.get_embedding(input)

        cnn_input = self.dropout(input_embeddings).unsqueeze(1)
        cnn_output = self.cnn_layer(cnn_input)
        cnn_output = funct.relu(cnn_output).squeeze(3)
        cnn_output = torch.transpose(cnn_output, 1, 2)
        cnn_output = torch.transpose(cnn_output, 0, 1)

        rnn_output, self.rnn_hidden_state = self.rnn_layer(cnn_output, self.rnn_hidden_state)
        rnn_output = torch.transpose(rnn_output, 0, 1)
        rnn_output = torch.transpose(rnn_output, 1, 2)
        rnn_output = funct.max_pool1d(rnn_output, rnn_output.size(2)).squeeze(2)

        dense_layer_1_output = self.dense_layer_1(rnn_output)
        dense_layer_2_output = self.dense_layer_2(funct.sigmoid(dense_layer_1_output))
        net_out = funct.sigmoid(dense_layer_2_output)
        return net_out


class ConditionPredictor(nn.Module):
    def forward(self, *input):
        pass

    def __init__(self, embedding_layer, args):
        super().__init__()
        self.args = args
        self.embedding_layer = embedding_layer


class SelectPredictor(nn.Module):
    def forward(self, *input):
        pass

    def __init__(self, embedding_layer, args):
        super().__init__()
        self.args = args
        self.embedding_layer = embedding_layer
