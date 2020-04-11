import pytreebank
import torch
from pytorch_transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from torch.utils.data import Dataset
import argparse
import os
import copy
import numpy as np


def right_pad(array, n=70):
    """Right padding."""
    current_len = len(array)
    if current_len > n:
        array[n - 1] = 102
        return array[: n]
    extra = n - current_len
    return array + ([0] * extra)


class SSTDataset(Dataset):
    """Configurable SST Dataset."""

    def __init__(self, dataset, split="train", max_length=70, bert="bert-base-uncased"):
        """Initializes the dataset with given configuration."""
        path = os.path.join(dataset, split + ".txt")
        tokenizer = BertTokenizer.from_pretrained(bert)
        self.max_length = max_length

        sentences = []
        labels = []
        with open(os.path.join(path), 'r', encoding='iso-8859-1') as f:
            for line in f.readlines():
                label = int(line[0])
                sentence = line[1:].strip()
                sentences.append(sentence)
                labels.append(label)

        self.data = [
            (
                right_pad(
                    tokenizer.encode("[CLS] " + sentence + " [SEP]"), self.max_length
                ),
                label,
            )
            for sentence, label in zip(sentences, labels)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x, y = self.data[index]
        x = torch.tensor(x)
        y = torch.tensor(y)
        return x, y


def train_one_epoch(model, optimizer, dataset, batch_size=32, device="cpu"):
    generator = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    model.train()
    train_loss, train_acc = 0.0, 0.0
    for batch, labels in generator:
        batch, labels = batch.to(device), labels.to(device)
        optimizer.zero_grad()
        loss, logits = model(batch, labels=labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred_labels = torch.argmax(logits, dim=1)
        train_acc += (pred_labels == labels).sum().item()
    train_loss /= len(dataset)
    train_acc /= len(dataset)
    return train_loss, train_acc


def evaluate_one_epoch(model, f_loss, dataset, batch_size=32, device="cpu"):
    generator = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    model.eval()
    loss, acc = 0.0, 0.0
    with torch.no_grad():
        for batch, labels in generator:
            batch, labels = batch.to(device), labels.to(device)
            logits = model(batch)[0]
            err = f_loss(logits, labels)
            loss += err.item()
            pred_labels = torch.argmax(logits, dim=1)
            acc += (pred_labels == labels).sum().item()
    loss /= len(dataset)
    acc /= len(dataset)
    return loss, acc


if __name__ == '__main__':
    """
    python bert.py -nc 5 -e 10 -dp "dataset/sst5"
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-dp", "--dataset_path", type=str, default="dataset/sst5",
                    help="path to the dataset, it has to contain 3 files: train.txt, dev.txt, test.txt")
    ap.add_argument("-ml", "--max_length", type=int, default=66,
                    help="max length for a sentence to consider,"
                         " if None it will correspond to the length of the longest sequence in the training set")
    ap.add_argument("-bs", "--batch_size", type=int, default=32,
                    help="batch size")
    ap.add_argument("-e", "--epochs", type=int, default=10,
                    help="number of training epochs")
    ap.add_argument("-nc", "--num_classes", type=int, default=5,
                    help="number of output classes")
    args = ap.parse_args()

    bert = "bert-base-uncased"
    epochs = args.epoch
    batch_size = args.batch_size
    max_length = args.max_length
    num_labels = args.num_classes
    dataset = args.dataset_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainset = SSTDataset(dataset, "train", max_length, bert)
    devset = SSTDataset(dataset, "dev", max_length, bert)
    testset = SSTDataset(dataset, "test", max_length, bert)

    model = BertForSequenceClassification.from_pretrained(bert, num_labels=num_labels).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    f_loss = torch.nn.CrossEntropyLoss()

    best_model = copy.deepcopy(model.state_dict())
    best_val_loss = np.inf

    for epoch in range(0, epochs):
        print("Epoch", epoch)

        train_loss, train_acc = train_one_epoch(model, optimizer, trainset, batch_size, device)
        print("Train loss: {}, Train accuracy: {}".format(train_loss, train_acc))

        val_loss, val_acc = evaluate_one_epoch(model, f_loss, devset, batch_size, device)
        print("Val loss: {}, Val accuracy: {}".format(val_loss, val_acc))

        if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model)
    test_loss, test_acc = evaluate_one_epoch(model, f_loss, testset, batch_size, device)
    print("Test loss: {}, Test accuracy: {}".format(test_loss, test_acc))
    print("Done!")
