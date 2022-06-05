import torch
import torch.nn as nn


class Net_encoder(nn.Module):
    def __init__(self, input_size, dim=64):
        super(Net_encoder, self).__init__()
        self.input_size = input_size
        self.dim = dim

        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, self.dim)

        )

    def forward(self, data):
        data = data.float().view(-1, self.input_size)
        embedding = self.encoder(data)

        return embedding


class Net_cell(nn.Module):
    def __init__(self, num_of_class, dim=64):
        super(Net_cell, self).__init__()
        self.cell = nn.Sequential(
            nn.Linear(dim, num_of_class)

        )

    def forward(self, embedding):
        cell_prediction = self.cell(embedding)

        return cell_prediction
