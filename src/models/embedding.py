import numpy as np
import torch
import torch.nn as nn

import constants.main_constants as const


class EmbeddingLayer(nn.Module):
    def forward(self, *input):
        pass

    def __init__(self, emb_size=const.EMBEDDING_SIZE, gpu=False, token_to_index=None, token_weights=None):
        super(EmbeddingLayer, self).__init__()
        self.gpu = gpu
        self.emb_size = emb_size
        self.token_to_index = token_to_index

        self.total_unique_tokens = len(token_to_index)
        self.embedding = nn.Embedding(self.total_unique_tokens, emb_size)
        self.embedding.weight.data.copy_(torch.from_numpy(token_weights.astype(np.float32)))

    def get_embedding(self, input):
        return self.embedding(input)

# TODO: Make embedding batch functions
