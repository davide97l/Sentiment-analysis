import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.data


class temporalvalue(nn.Module):
    def __init__(self, kernelsize, maxlen):
        super(temporalvalue, self).__init__()
        self.kernelsize = kernelsize
        self.maxlen = maxlen
        self.totallen = self.maxlen - self.kernelsize + 1

    def forward(self, value):
        semi_filters = []
        for i in range(self.totallen):
            semi_filters.append(torch.unsqueeze(value[:, i:i + self.kernelsize, :], 1))

        semifilters = torch.cat(semi_filters, 1)
        return semifilters


class timedistributed(nn.Module):
    def __init__(self, hiddendim, embeddim, drop):
        super(timedistributed, self).__init__()
        self.hiddendim = hiddendim
        self.embeddim = embeddim
        self.drop = drop
        self.drop_inp = nn.Dropout(self.drop)
        self.lstm = nn.LSTM(self.embeddim, self.hiddendim, batch_first=True)

    def forward(self, x):
        timesteps = []
        for i in range(x.size(1)):
            inp = self.drop_inp(x[:, i, :, :])
            out, _ = self.lstm(inp, None)
            # print(out.shape)
            timesteps.append(torch.unsqueeze(out[:, -1, :], 1))
        finalout = torch.cat(timesteps, 1)
        return finalout


class RNF(nn.Module):
    """Recurrent neural filter convolutional model"""
    def __init__(self, embed_dim, embed, kernel_size=5, max_len=50, hidden_dim=300,
                 embed_drop=0.2, drop=0.2, num_classes=5):
        super().__init__()

        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.embed_drop = embed_drop
        self.drop = drop
        self.num_classes = num_classes

        self.embed_layer = embed

        self.temporal = temporalvalue(self.kernel_size, self.max_len)
        self.timedist = timedistributed(self.hidden_dim, self.embed_dim, self.drop)
        self.dense = nn.Linear(self.max_len - self.kernel_size + 1, self.num_classes)

        self.drop_embed = nn.Dropout(self.embed_drop)
        self.drop_inp = nn.Dropout(self.drop)

    def forward(self, x):
        embedout = self.embed_layer(x)
        embedout = self.drop_embed(embedout)

        cnnrnf = self.temporal(embedout)
        cnnrnf = self.timedist(cnnrnf)
        cnnrnf = torch.max(cnnrnf, 2)[0]
        cnnrnf = self.dense(cnnrnf)

        return cnnrnf


class CNN(nn.Module):
    """Convolutional model"""
    def __init__(self, embedding, embedding_dim, n_filters=3, filter_sizes=(2, 3, 4), output_dim=5,
                 dropout=0, embedding_dropout=0):
        super().__init__()

        self.embedding = embedding

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.embedding_dropout = nn.Dropout(embedding_dropout)

    def forward(self, x):
        # text = [batch size, sent len]
        embedded = self.embedding_dropout(self.embedding(x))
        # embedded = [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)
        # embedded = [batch size, 1, sent len, emb dim]
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batch size, n_filters * len(filter_sizes)]
        logits = self.fc(cat)
        # logits = [n_filters * len(filter_sizes)]
        return logits


class RNN(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout):
        super().__init__()

        self.embedding = embedding

        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        # text = [sent len, batch size]

        embedded = self.dropout(self.embedding(text))

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
        return self.fc(hidden)


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
