import torch
import torch.utils
import torch.utils.data
import torch.nn.functional as F


def batch2RNNinput(x_batch):
    """Adapt batch  such to be ready to packed during RNN forward pass"""
    x_batch = x_batch.sort(axis=0)[0]
    x_batch = torch.flip(x_batch, dims=[1, 0])
    x_batch = x_batch.T
    x_data = x_batch[1:]
    x_lengths = x_batch[:1].squeeze(0)

    return x_data, x_lengths


def evaluate(model, eval_set, device, mode):
    model.eval()
    with torch.no_grad():
        tot_loss = 0.0
        tot_accuracy = 0.0
        for x, y in eval_set:
            x = x.long().to(device)
            y = y.long().to(device)

            if mode == 'rnn':
                x_data, x_len = batch2RNNinput(x)
                if 0. in x_len:
                    continue
                out = model(x_data, x_len)
            else:
                out = model(x)

            tot_loss += F.cross_entropy(out, y, reduction='sum').item()

            predictions = torch.max(out, 1)[1]
            tot_accuracy += torch.sum(predictions == y).item()

        loss = tot_loss / len(eval_set.dataset)
        acc = tot_accuracy / len(eval_set.dataset)
        return loss, acc


def train(model, train_set, optimizer, device, mode):
    model.train()
    tot_loss = 0.0
    tot_accuracy = 0.0
    for x, y in train_set:
        x = x.long().to(device)
        y = y.long().to(device)

        model.zero_grad()

        if mode == 'rnn':
            x_data, x_len = batch2RNNinput(x)
            if 0. in x_len:
                continue
            out = model(x_data, x_len)
        else:
            out = model(x)

        loss = F.cross_entropy(out, y, reduction='sum')
        tot_loss += loss.item()

        predictions = torch.max(out, 1)[1]
        tot_accuracy += torch.sum(predictions == y).item()

        loss.backward()
        optimizer.step()

    loss = tot_loss / len(train_set.dataset)
    acc = tot_accuracy / len(train_set.dataset)
    return loss, acc
