from loader import *
from vocabulary import *
from model import *
from train import *
import argparse
import copy
import numpy as np


if __name__ == '__main__':
    """
        python main.py -e 10 -es 300 -ed 0.2 -dr 0.2
        python main.py -ml 50 -mc 3 -nf 20 -bs 64 -dr 0.2 -ed 0.2 -es 200 -fs "2,3,4" -e 10 -m "cnn" -dp "dataset/sst5" -nc 5
        python main.py --help
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-dp", "--dataset_path", type=str, default="dataset/sst5",
                    help="path to the dataset, it has to contain 3 files: train.txt, dev.txt, test.txt")
    ap.add_argument("-ep", "--embedding_path", type=str, default=None,
                    help="path to the embed file, if None a new embedding will be learned form scratch")
    ap.add_argument("-es", "--embedding_size", type=int, default=300,
                    help="embedding size (you don't need to set it if loading an existing embedding)")
    ap.add_argument("-ed", "--embedding_dropout", type=float, default=0,
                    help="embedding layer dropout probability value (0 = no dropout)")
    ap.add_argument("-dr", "--dropout", type=float, default=0,
                    help="dropout probability value (0 = no dropout)")
    ap.add_argument("-ml", "--max_length", type=int, default=None,
                    help="max length for a sentence to consider,"
                         " if None it will correspond to the length of the longest sequence in the training set")
    ap.add_argument("-bs", "--batch_size", type=int, default=64,
                    help="batch size")
    ap.add_argument("-e", "--epochs", type=int, default=10,
                    help="number of training epochs")
    ap.add_argument("-nc", "--num_classes", type=int, default=5,
                    help="number of output classes")
    ap.add_argument("-mc", "--min_count", type=int, default=2,
                    help="remove rare words that appear in total less than min_count times")
    ap.add_argument("-ps", "--padding_side", type=str, default="right",
                    help="add padding at the 'left' or at the 'right' of the sentence")
    ap.add_argument("-nf", "--n_filters", type=int, default=100,
                    help="number of convolutional filters")
    ap.add_argument("-fs", "--filter_sizes", type=str, default="2,3,4",
                    help="size of the filters")
    ap.add_argument("-sd", "--save_directory", type=str, default="model",
                    help="directory where to save a trained model")
    ap.add_argument("-lp", "--load_path", type=str, default=None,
                    help="path to load a trained model, if None create a new model")
    ap.add_argument("-m", "--model", type=str, default="cnn",
                    help="type of model: 'cnn', 'rnf'")
    ap.add_argument("-hs", "--hidden_size", type=int, default=300,
                    help="hidden layers size")
    ap.add_argument("-nl", "--n_layers", type=int, default=2,
                    help="number of hidden layers")
    ap.add_argument("-bi", "--bidirectional", default=False, action='store_true',
                    help="if True use bidirectional recurrent cells")
    args = ap.parse_args()

    model_type = args.model

    dataset_path = args.dataset_path
    embedding_path = args.embedding_path
    save_path = args.save_directory
    full_save_path = os.path.join(save_path, model_type)
    if not os.path.exists(full_save_path):
        os.makedirs(full_save_path)
    full_save_path += ".tar"
    load_path = args.load_path
    if load_path:
        checkpoint = torch.load(load_path)

    embedding_dropout = args.embedding_dropout
    dropout = args.dropout
    max_len = args.max_length
    batch_size = args.batch_size
    epochs = args.epochs
    num_classes = args.num_classes
    embedding_dim = args.embedding_size
    min_count = args.min_count
    padding = args.padding_side
    # only for CNN
    n_filters = args.n_filters
    filter_sizes = list(map(int, args.filter_sizes.strip().split(",")))
    # only for RNN
    hidden_size = args.hidden_size
    num_layers = args.n_layers
    bidirectional = args.bidirectional

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    print("Loading dataset...")
    x_train, x_val, x_test, y_train, y_val, y_test = load_data(dataset_path)

    print("Preprocessing dataset...")
    x_train, x_val, x_test = preprocess_data(x_train), preprocess_data(x_val), preprocess_data(x_test)

    # convert labels to vectors and put on device
    y_train = torch.from_numpy(np.asarray(y_train, dtype='int32')).to(device)
    y_val = torch.from_numpy(np.asarray(y_val, dtype='int32')).to(device)
    y_test = torch.from_numpy(np.asarray(y_test, dtype='int32')).to(device)

    print("Creating vocabulary...")
    vocabulary = Voc()
    vocabulary.addSentences(x_train)
    vocabulary.addSentences(x_val)
    vocabulary.addSentences(x_test)
    print("Vocabulary contains " + str(vocabulary.num_words) + " words")

    # remove rare words
    if min_count > 1:
        print("Trimming vocabulary...")
        vocabulary.trim(min_count)
        x_train = vocabulary.fix_sentences(x_train)
        x_val = vocabulary.fix_sentences(x_val)
        x_test = vocabulary.fix_sentences(x_test)
        print("Trimmed to " + str(vocabulary.num_words) + " words")

    if max_len is None:
        # set maximum sentence length to the length of the longest sentence in the train set
        max_len = len(max(x_train, key=len))

    print(len(vocabulary.get_words()))

    if embedding_path:
        print("Loading embedding...")
        # load embedding from path
        embedding2index, embedding_dim = load_embed(embedding_path)
        # get embedding matrix
        embedding_matrix = load_embedding_matrix(embedding2index, vocabulary.get_words(), embedding_dim)
        embedding = nn.Embedding.from_pretrained(embedding_matrix, padding_idx=vocabulary.PAD_token)
    else:
        print("Initializing new embedding...")
        embedding = nn.Embedding(vocabulary.num_words, embedding_dim, padding_idx=vocabulary.PAD_token)
    embedding.to(device)

    # convert input sentences to their indices and pad them
    x_train_idx = vocabulary.pad_sentences(x_train, max_len, padding)
    x_val_idx = vocabulary.pad_sentences(x_val, max_len, padding)
    x_test_idx = vocabulary.pad_sentences(x_test, max_len, padding)

    # append its length to each sentence
    if model_type == "rnn":
        for i, line in enumerate(x_train_idx):
            line.append(sum(1 for x in line if x != 0))
            x_train_idx[i] = line
        for i, line in enumerate(x_val_idx):
            line.append(sum(1 for x in line if x != 0))
            x_val_idx[i] = line
        for i, line in enumerate(x_test_idx):
            line.append(sum(1 for x in line if x != 0))
            x_test_idx[i] = line

    print("Building model...")
    if model_type == 'rnf':
        model = RNF(embedding_dim, embedding, filter_sizes[0], max_len, hidden_size,
                    embedding_dropout, dropout, num_classes).to(device)
    if model_type == 'cnn':
        model = CNN(embedding, embedding_dim, n_filters, filter_sizes, num_classes,
                    dropout, embedding_dropout).to(device)
    if model_type == 'rnn':
        model = RNN(embedding, embedding_dim, hidden_size, num_classes, num_layers, bidirectional, dropout).to(device)

    if load_path:
        print("Restoring saved model")
        model.load_state_dict(checkpoint['model'])

    # convert sequences of indexes to tensors and put on device
    x_train_idx = torch.from_numpy(np.asarray(x_train_idx, dtype='int32')).to(device)
    x_val_idx = torch.from_numpy(np.asarray(x_val_idx, dtype='int32')).to(device)
    x_test_idx = torch.from_numpy(np.asarray(x_test_idx, dtype='int32')).to(device)

    x_train, x_val, x_test = load_datasets(x_train_idx, x_val_idx, x_test_idx, y_train, y_val, y_test,
                                           batch_size)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    print("Begin training...")

    best_valid_loss = float('inf')
    best_model_dict = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):

        train_loss, train_acc = train(model, x_train, optimizer, criterion, device, model_type)
        valid_loss, valid_acc = evaluate(model, x_val, criterion, device, model_type)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model_dict = copy.deepcopy(model.state_dict())

        print(f'Epoch: {epoch+1:02}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\tVal. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

    print("Evaluation...")
    model.load_state_dict(best_model_dict)
    test_loss, test_acc = evaluate(model, x_test, criterion, device, model_type)
    print('Test acc: {:.3f}%'.format(test_acc*100))

    print("Saving model...")
    torch.save({
        'model': model.state_dict(),
    }, os.path.join(full_save_path))
    print("Model saved in: " + str(full_save_path))

    print("Program terminated")
