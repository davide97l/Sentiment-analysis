# Sentiment-analysis

Opinion mining (sometimes known as sentiment analysis or emotion AI) refers to the use of natural language processing, text analysis, computational linguistics, and biometrics to systematically identify, extract, quantify, and study aﬀective states and subjective information. Sentiment analysis is widely applied to the voice of the customer materials such as reviews and survey responses, online and social media, and healthcare materials for applications that range from marketing to customer service to clinical medicine. Industrial circles utilize opinion mining techniques to detect people’s preference for further recommendation, such as movie reviews and restaurant reviews. In this assignment, we need to establish a sentiment classiﬁcation model for the given sentence. In this project I have implemented a CNN, DCNN, RNN and RNF model for sentence sentiment classiﬁcation.

## Dataset preparation
Each dataset should be formed by 3 files: `train.txt`, `dev.txt`, `test.txt`, each having the following structure. Make sure the 3 files are placed the same folder.
```
  l1 sentence1
  l2 sentence2
  ...
  lN sentenceN
```
You can find some examples in the directories `dataset/sst5` and `dataset/sst2` taken from Stanford Sentiment Treebank dataset with 5 and 2 classes respectively.

## Data preprocessing
During the preprocessing phase the following pipeline will be applied to each sentence in each set: 
- Removal of punctuation.
- Characters Lower case conversion.
- Filtering of non-purely alphabetical words.
- Filtering of stop words.
- Filtering of words less than two characters long.

For more details about the preprocessign phase you can refer to the `loader.py` file.

## Pretrained embedding
You can download a pretrained embadding such as Glove (https://nlp.stanford.edu/projects/glove/) to initialize your embedding matrix. Once downloaded, put it in the folder `embed` (ex: `embed/glove6B.300d.txt`).

## Pretrained models
I have uploaded some pretrained models in the folder `model`, one for each of the three models. All the models have been trained for 20 epochs, initialized with the glove.6B.300d embedding and with almost all the other parameters left as their default value (non-default parameter values are specified in the appropriate command). When loading a pretrained model is it important that the used parameters are the same as the parameters the model as been trained on. You can see the command to load each of the pretrained model in their relative section. After training, a model will be saved at the path `model/model_type/model.tar` which it can be changed setting the appropriate parameters (see `--help`). Moreover, when using a pretrained model you should set the flag `--no_training` so to avoid training it again.

## CNN (Convolutional network)

**Note:** CNN model consists of multiple filters of different sizes which will look at different n-grams (1xn filters) along a sentence and learn to select the most useful ones in order to classify it.

Train a convolutional network.
```
  python main.py -m "cnn" -dp "dataset/sst5"
```
Train a convolutional network with dropout and embedding dropout.
```
  python main.py -m "cnn" -dp "dataset/sst5" -dr 0.4 -ed 0.4
```
Add more filters with different sizes and change their depth (number of output channels).
```
  python main.py -m "cnn" -dp "dataset/sst5" -fs "2,3,4,5" -nf 50
```
Train it with pretrained embedding.
```
  python main.py -m "cnn" -dp "dataset/sst5" -ep "embed/glove6B.300d.txt"
```
Train on a dataset with binary labels.
```
  python main.py -m "cnn" -dp "dataset/sst2" -nc 2
```
Load a pretrained model.
```
  python main.py -m "cnn" -dp "dataset/sst5" -lp "model/cnn/cnn_model.tar" -dr 0.4 -ed 0.4 --no_training
```
Finally, you can use the command `--help` to visualize the full list of parameters you can fine tune as well as their description.

## RNN (Recurrent network)

**Note:** RNN cells consists of bidirectional LSTM cells, it can also be set the number of layers.

Train a recurrent network.
```
  python main.py -m "rnn" -dp "dataset/sst5"
```
Change the number of layers and hidden size.
```
  python main.py -m "rnn" -dp "dataset/sst5" -nl 2 -hs 100
```
Load a pretrained model (non-default parameter: `-nl 1`, `-hs 100`).
```
  python main.py -m "rnn" -dp "dataset/sst5" -lp "model/rnn/rnn_model.tar" -dr 0.4 -ed 0.4 -bi -hs 100 --no_training
```

## RNF (Recurrent neural filter)

**Note:** This model has been implemented based on the following paper: https://arxiv.org/pdf/1808.09315v1.pdf

Train a convolutional network with recurrent neural filter.
```
  python main.py -m "rnf" -dp "dataset/sst5"
```
Change the size of the recurrent filter
```
  python main.py -m "rnf" -dp "dataset/sst5" -rnfs 10
```
Load a pretrained model.
```
  python main.py -m "rnf" -dp "dataset/sst5" -lp "model/rnf/rnf_model.tar" -dr 0.4 -ed 0.4 --no_training
```

## DCNN (Deep convolutional network)

**Note:** DCNN model is based on the CNN architecture with a deep dense network built on top of it. Thus, the only new parameter that is introduced is `n_layers` (used also by RNN) which in this case specify the number of dense hidden layers.

Train a deep convolutional network.
```
  python main.py -m "dcnn" -dp "dataset/sst5"
```
Change the number of dense hidden layers.
```
  python main.py -m "dcnn" -dp "dataset/sst5" -nl 2
```
Load a pretrained model (non-default parameter: `-nl 1`).
```
  python main.py -m "cnn" -dp "dataset/sst5" -lp "model/dcnn/dcnn_model.tar" -dr 0.4 -ed 0.4 -nl 1 --no_training
```
Finally, you can use the command `--help` to visualize the full list of parameters you can fine tune as well as their description.


## References

- https://github.com/bentrevett/pytorch-sentiment-analysis
- https://github.com/avinashsai/Recurrent-Neural-Filters
- https://github.com/bloomberg/cnn-rnf
- https://arxiv.org/pdf/1808.09315v1.pdf
