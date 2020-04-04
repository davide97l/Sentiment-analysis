from loader import *
from vocabulary import *
from model import *
from train import *
import copy
import numpy as np
import argparse


if __name__ == '__main__':
    """
        python main.py -e 10 -es 300 -ed 0.2 -dr 0.2
        python main.py -ml 50 -mc 2 -nf 100 -bs 32 -dr 0.4 -ed 0.4 -es 300 -fs "1,2,3,5" -e 10 -m "cnn"
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
    ap.add_argument("-bs", "--batch_size", type=int, default=32,
                    help="batch size")
    ap.add_argument("-e", "--epochs", type=int, default=10,
                    help="number of training epochs")
    ap.add_argument("-nc", "--num_classes", type=int, default=5,
                    help="number of output classes")
    ap.add_argument("-mc", "--min_count", type=int, default=1,
                    help="remove rare words that appear in total less than min_count times")
    ap.add_argument("-sd", "--save_directory", type=str, default="model",
                    help="directory where to save a trained model")
    ap.add_argument("-mn", "--model_name", type=str, default="model",
                    help="used to distinguish a model when saving")
    ap.add_argument("-lp", "--load_path", type=str, default=None,
                    help="path to load a trained model, if None create a new model")
    ap.add_argument("-m", "--model", type=str, default="cnn",
                    help="type of model: 'cnn', 'rnn', 'rnf', 'dcnn'")
    ap.add_argument("-hs", "--hidden_size", type=int, default=300,
                    help="hidden layers size (rnn, rnf)")
    ap.add_argument("-nl", "--n_layers", type=int, default=2,
                    help="number of hidden layers (rnn, dcnn)")
    ap.add_argument("-bi", "--bidirectional", default=False, action='store_true',
                    help="if True use bidirectional recurrent cells (rnn)")
    ap.add_argument("-nf", "--n_filters", type=int, default=100,
                    help="number of convolutional filters (cnn, dcnn)")
    ap.add_argument("-fs", "--filter_sizes", type=str, default="2,3,4",
                    help="size of the filters, if 'rnf' it will only used the first element (cnn, rnf, dcnn)")
    ap.add_argument("-rnfs", "--rnf_size", type=int, default=5,
                    help="size of the recurrent filter (rnf)")
    ap.add_argument("-nt", "--no_training", default=False, action='store_true',
                    help="skip the training phase (useful if you just want to evaluate your model)")
    args = ap.parse_args()

    model_type = args.model
    model_name = args.model_name
    no_training = args.no_training  # skip training phase

    # directories
    dataset_path = args.dataset_path
    embedding_path = args.embedding_path
    save_path = args.save_directory
    full_save_path = os.path.join(save_path, model_type)
    if not os.path.exists(full_save_path):
        os.makedirs(full_save_path)
    full_save_path = os.path.join(save_path, model_type, model_name + ".tar")
    load_path = args.load_path
    if load_path is not None:
        checkpoint = torch.load(load_path)

    # general parameters
    embedding_dropout = args.embedding_dropout
    dropout = args.dropout
    max_len = args.max_length
    batch_size = args.batch_size
    epochs = args.epochs
    num_classes = args.num_classes
    embedding_dim = args.embedding_size
    min_count = args.min_count
    # only for CNN
    n_filters = args.n_filters
    filter_sizes = list(map(int, args.filter_sizes.strip().split(",")))
    # only for RNN
    hidden_size = args.hidden_size
    num_layers = args.n_layers
    bidirectional = args.bidirectional
    # only for RNF
    rnf_size = args.rnf_size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Detected device:", device)

    print("Loading dataset...")
    x_train, x_val, x_test, y_train, y_val, y_test = load_data(dataset_path)
    print("Preprocessing dataset...")
    x_train, x_val, x_test = preprocess_data(x_train), preprocess_data(x_val), preprocess_data(x_test)

    print("Creating vocabulary...")
    vocabulary = Voc()
    vocabulary.addSentences(x_train)
    print("Vocabulary contains " + str(vocabulary.num_words) + " words")
    # remove rare words
    if min_count > 1:
        print("Trimming vocabulary...")
        vocabulary.trim(min_count)
        x_train = vocabulary.fix_sentences(x_train)
        print("Trimmed to " + str(vocabulary.num_words) + " words")

    if max_len is None:
        # set maximum sentence length to the length of the longest sentence in the train set
        max_len = len(max(x_train, key=len))

    # convert input sentences to their indices and pad them
    x_train_idx = vocabulary.pad_sentences(x_train, max_len, 'left')
    x_val_idx = vocabulary.pad_sentences(x_val, max_len, 'left')
    x_test_idx = vocabulary.pad_sentences(x_test, max_len, 'left')

    # append its length to each sentence (only used for RNN)
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

    embedding_matrix = None
    if embedding_path is not None and load_path is None:
        print("Loading embedding...")
        # load embedding from path
        embedding2index, embedding_size = load_embedding(embedding_path)
        # get embedding matrix corresponding to the words in vocabulary
        embedding_matrix = load_embedding_matrix(embedding2index, vocabulary.get_words(), embedding_size)

    print("Building model...")
    if model_type == 'rnf':
        model = RNF(embedding_matrix, embedding_dim, rnf_size, max_len, hidden_size,
                    embedding_dropout, dropout, num_classes, vocabulary.num_words).to(device)
    elif model_type == 'cnn':
        model = CNN(embedding_matrix, embedding_dim, n_filters, filter_sizes,
                    embedding_dropout, dropout, num_classes, vocabulary.num_words).to(device)
    elif model_type == 'dcnn':
        model = DCNN(embedding_matrix, embedding_dim, n_filters, filter_sizes,
                     embedding_dropout, dropout, num_classes, num_layers, vocabulary.num_words).to(device)
    elif model_type == 'rnn':
        model = RNN(embedding_matrix, embedding_dim, hidden_size, num_layers, bidirectional,
                    embedding_dropout, dropout, num_classes, vocabulary.num_words).to(device)
    else:
        raise Exception("This model doesn't exists, supported models: 'cnn', 'dcnn', 'rnn', 'rnf'")
    print("Using model: ", model_name)

    if load_path:
        print("Restoring saved model...")
        model.load_state_dict(checkpoint['model'])

    x_train, x_val, x_test = get_loaders(x_train_idx, x_val_idx, x_test_idx, y_train, y_val, y_test,
                                         batch_size, device)

    if not no_training:
        optimizer = torch.optim.Adam(model.parameters())
        best_model = copy.deepcopy(model.state_dict())
        best_val_loss = np.inf
        print("Start training")
        for epoch in range(epochs):

            train_loss, train_acc = train(model, x_train, optimizer, device, model_type)
            val_loss, val_acc = evaluate(model, x_val, device, model_type)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(model.state_dict())

            print("Epoch {}".format(epoch))
            print("\tTrain Loss {:.3f} | Train Acc {:.3f}%".format(train_loss, train_acc))
            print("\tVal Loss {:.3f} | Val Acc {:.3f}%".format(val_loss, val_acc))

        model.load_state_dict(best_model)

    _, test_acc = evaluate(model, x_test, device, model_type)
    print('Test acc: {:.3f}%'.format(test_acc))

    if not no_training:
        print("Saving model...")
        torch.save({
            'model': model.state_dict(),
        }, os.path.join(full_save_path))
        print("Model saved in: " + str(full_save_path))

    print("Program terminated")
