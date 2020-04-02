#nltk.download('stopwords')
from nltk.corpus import stopwords
import string
import os
from torch.distributions import uniform
import torch
import numpy as np


def preprocess_data(lines):
    """
    Split tokens on white space.
    Remove all punctuation from words.
    Remove all words that are not purely comprised of alphabetical characters.
    Remove all words that are known stop words.
    Remove all words that have a length <= 1 character.
    """
    formatted_lines = []
    for line in lines:
        # split into tokens by white space
        tokens = line.split()
        # remove punctuation from each token and convert to lowercase
        table = str.maketrans('', '', string.punctuation)
        tokens = [w.translate(table).lower() for w in tokens]
        # remove remaining tokens that are not alphabetic
        tokens = [word for word in tokens if word.isalpha()]
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        tokens = [w for w in tokens if w not in stop_words]
        # filter out short tokens
        tokens = [word for word in tokens if len(word) > 1]
        # join the new tokens to form the formatted sentence
        sentence = " ".join(tokens)
        formatted_lines.append(sentence)
    return formatted_lines


def load_data(dataset_path):
    """Load test, validation and train data with their labels"""
    x_train = []
    y_train = []
    x_val = []
    y_val = []
    x_test = []
    y_test = []

    with open(os.path.join(dataset_path, "train.txt"), 'r', encoding='iso-8859-1') as f:
        for line in f.readlines():
            label = int(line[0])
            sentence = line[1:].strip()
            x_train.append(sentence)
            y_train.append(label)
    with open(os.path.join(dataset_path, "dev.txt"), 'r', encoding='iso-8859-1') as f:
        for line in f.readlines():
            label = int(line[0])
            sentence = line[1:].strip()
            x_val.append(sentence)
            y_val.append(label)
    with open(os.path.join(dataset_path, "test.txt"), 'r', encoding='iso-8859-1') as f:
        for line in f.readlines():
            label = int(line[0])
            sentence = line[1:].strip()
            x_test.append(sentence)
            y_test.append(label)

    return x_train, x_val, x_test, y_train, y_val, y_test


def load_embedding(embed_path):
    """Load word embedding and return word-embedding vocabulary"""
    embedding2index = {}
    with open(embed_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            lexicons = line.split()
            word = lexicons[0]
            embedding2index[word] = torch.from_numpy(np.asarray(lexicons[1:], dtype='float32'))
        embedding_size = len(lexicons) - 1
    return embedding2index, embedding_size


def load_embedding_matrix(embedding, words, embedding_size):
    """Add new words in the embedding matrix and return it"""
    embedding_matrix = torch.zeros(len(words), embedding_size)
    for i, word in enumerate(words):
        # Note: PAD embedded as sequence of zeros
        if word not in embedding:
            if word != 'PAD':
                embedding_matrix[i] = uniform.Uniform(-0.25, 0.25).sample(torch.Size([embedding_size]))
        else:
            embedding_matrix[i] = embedding[word]
    return embedding_matrix


def get_loaders(x_train, x_val, x_test, y_train, y_val, y_test, batch_size, device):
    """Return iterables over train, validation and test dataset"""

    # convert labels to vectors and put on device
    y_train = torch.from_numpy(np.asarray(y_train, dtype='int32')).to(device)
    y_val = torch.from_numpy(np.asarray(y_val, dtype='int32')).to(device)
    y_test = torch.from_numpy(np.asarray(y_test, dtype='int32')).to(device)

    # convert sequences of indexes to tensors and put on device
    x_train = torch.from_numpy(np.asarray(x_train, dtype='int32')).to(device)
    x_val = torch.from_numpy(np.asarray(x_val, dtype='int32')).to(device)
    x_test = torch.from_numpy(np.asarray(x_test, dtype='int32')).to(device)

    train_array = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_array, batch_size)

    val_array = torch.utils.data.TensorDataset(x_val, y_val)
    val_loader = torch.utils.data.DataLoader(val_array, batch_size)

    test_array = torch.utils.data.TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_array, batch_size)

    return train_loader, val_loader, test_loader
