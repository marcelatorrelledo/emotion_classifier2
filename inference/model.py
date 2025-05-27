import torch.nn as nn
import torch

class MyModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, pad_idx, embedding_matrix=None, freeze_embeddings=True, n_layers=2, dropout=0.3):
        super(MyModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(embedding_matrix)
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if n_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        h = torch.cat((hidden[-2], hidden[-1]), dim=1)
        h = self.dropout(h)
        return self.fc(h)
