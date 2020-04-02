import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.data


class temporal_value(nn.Module):
    def __init__(self, kernel_size, max_len):
        super().__init__()
        self.kernel_size = kernel_size
        self.max_len = max_len
        self.total_len = self.max_len - self.kernel_size + 1

    def forward(self, value):
        semi_filters = []
        for i in range(self.total_len):
            semi_filters.append(torch.unsqueeze(value[:, i:i + self.kernel_size, :], 1))

        semi_filters = torch.cat(semi_filters, 1)
        return semi_filters


class time_distributed(nn.Module):
    def __init__(self, hidden_dim, embedding_dim, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        timesteps = []
        for i in range(x.size(1)):
            inp = self.dropout(x[:, i, :, :])
            out, _ = self.lstm(inp, None)
            # print(out.shape)
            timesteps.append(torch.unsqueeze(out[:, -1, :], 1))
        return torch.cat(timesteps, 1)


class RNF(nn.Module):
    """Recurrent neural filter convolutional model"""
    def __init__(self, embedding_matrix, embedding_dim, kernel_size=5, max_len=50, hidden_dim=300,
                 embedding_dropout=0, dropout=0, num_classes=5, vocab_size=None):
        super().__init__()

        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix, padding_idx=0)
        else:
            if vocab_size is None:
                raise Exception("Could not create new embedding: missing vocabulary size!")
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.temporal = temporal_value(kernel_size, max_len)
        self.time_dist = time_distributed(hidden_dim, embedding_dim, dropout)
        self.fc = nn.Linear(max_len - kernel_size + 1, num_classes)

        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedout = self.embedding_dropout(self.embedding(x))

        cnnrnf = self.temporal(embedout)
        cnnrnf = self.time_dist(cnnrnf)
        cnnrnf = torch.max(cnnrnf, 2)[0]
        cnnrnf = self.fc(cnnrnf)

        return cnnrnf


class CNN(nn.Module):
    """Convolutional model"""
    def __init__(self, embedding_matrix, embedding_dim, n_filters=3, filter_sizes=(2, 3, 4), embedding_dropout=0,
                 dropout=0, output_dim=5, vocab_size=None):
        super().__init__()

        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix, padding_idx=0)
        else:
            if vocab_size is None:
                raise Exception("Could not create new embedding: missing vocabulary size!")
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.conv_layers = nn.ModuleList([nn.Conv2d(in_channels=1,
                                          out_channels=n_filters,
                                          kernel_size=(fs, embedding_dim)) for fs in filter_sizes])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.embedding_dropout = nn.Dropout(embedding_dropout)

    def forward(self, x):
        # text = [batch size, sent len]
        embedded = self.embedding_dropout(self.embedding(x))
        # embedded = [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)
        # embedded = [batch size, 1, sent len, emb dim]
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.conv_layers]
        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batch size, n_filters * len(filter_sizes)]
        logits = self.fc(cat)
        # logits = [n_filters * len(filter_sizes)]
        return logits


class RNN(nn.Module):
    def __init__(self, embedding_matrix, embedding_dim, hidden_dim=300, n_layers=2, bidirectional=True,
                 embedding_dropout=0, dropout=0, output_dim=5, vocab_size=None):
        super().__init__()

        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix, padding_idx=0)
        else:
            if vocab_size is None:
                raise Exception("Could not create new embedding: missing vocabulary size!")
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.embedding_dropout = nn.Dropout(embedding_dropout)

    def forward(self, text, text_lengths):
        # text = [sent len, batch size]

        embedded = self.embedding_dropout(self.embedding(text))

        # embedded = [sent len, batch size, emb dim]

        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)

        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        # output = [sent len, batch size, hid dim * num directions]
        # output over padding tokens are zero tensors

        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        # hidden = [batch size, hid dim * num directions]
        return self.fc(hidden).squeeze(1)


if __name__ == '__main__':
    # Testing Model
    print("Testing CNN model...")
    x = torch.ones(4, 10).long()
    sentence = ["This is just a test to check whether the model work"]
    embedding = nn.Embedding(11, 10)
    model = CNN(embedding, 10)
    out = model(x)
    assert out.shape == torch.Size([4, 5])
    print("Test passed!")
