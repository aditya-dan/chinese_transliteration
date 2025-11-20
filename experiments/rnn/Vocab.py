class Vocab:
    def __init__(self, stoi):
        self.stoi = stoi
        self.itos = {i: s for s, i in stoi.items()}
