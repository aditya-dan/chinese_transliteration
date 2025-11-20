import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, output_size, emb_size, hidden_size):
        super().__init__()

        self.embedding = nn.Embedding(output_size, emb_size)

        self.gru = nn.GRU(
            input_size=emb_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, tgt, hidden):
        embedded = self.embedding(tgt)
        output, hidden = self.gru(embedded, hidden)
        output = self.fc_out(output.squeeze(1))
        return output, hidden
