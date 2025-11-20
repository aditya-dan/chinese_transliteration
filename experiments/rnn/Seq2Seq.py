import torch
from torch import nn


def translate(model, src_sentence, src_tokenizer, src_vocab, tgt_vocab, max_len=50, device="cpu"):

    model.eval()

    tokens = src_tokenizer(src_sentence)

    tokens = ["<sos>"] + tokens + ["<eos>"]

    src_indices = [src_vocab.stoi.get(tok, src_vocab.stoi["<unk>"]) for tok in tokens]

    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)
    src_lengths = torch.LongTensor([len(src_indices)])

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_lengths)

    tgt_index = tgt_vocab.stoi["<sos>"]
    tgt_tensor = torch.LongTensor([[tgt_index]]).to(device)

    translated_tokens = []

    for _ in range(max_len):
        with torch.no_grad():
            output, hidden = model.decoder(tgt_tensor, hidden)
        next_tok = output.argmax(1).item()

        if next_tok == tgt_vocab.stoi["<eos>"]:
            break

        translated_tokens.append(tgt_vocab.itos[next_tok])

        tgt_tensor = torch.LongTensor([[next_tok]]).to(device)

    return " ".join(translated_tokens)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_lengths, tgt, teacher_forcing_ratio=0.5):
        batch_size = tgt.size(0)
        tgt_len = tgt.size(1)
        vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(self.device)

        encoder_outputs, hidden = self.encoder(src, src_lengths)

        input_tok = tgt[:, 0].unsqueeze(1)

        for t in range(1, tgt_len):
            output, hidden = self.decoder(input_tok, hidden)
            outputs[:, t] = output

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1).unsqueeze(1)

            input_tok = tgt[:, t].unsqueeze(1) if teacher_force else top1

        return outputs
