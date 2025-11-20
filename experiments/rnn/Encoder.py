import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, dropout=0.0):
        super().__init__()

        self.embedding = nn.Embedding(input_size, emb_size)

        self.gru = nn.GRU(
            input_size=emb_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=dropout
        )

        self.num_directions = 2
        self.hidden_size = hidden_size

    def forward(self, src, src_lengths=None):

        embedded = self.embedding(src)

        if src_lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        outputs, hidden = self.gru(embedded)

        if src_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        return outputs, hidden
