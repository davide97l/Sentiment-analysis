# Sentiment-analysis

Opinion mining (sometimes known as sentiment analysis or emotion AI) refers to the use of natural language processing, text analysis, computational linguistics, and biometrics to systematically identify, extract, quantify, and study aﬀective states and subjective information. Sentiment analysis is widely applied to the voice of the customer materials such as reviews and survey responses, online and social media, and healthcare materials for applications that range from marketing to customer service to clinical medicine. Industrial circles utilize opinion mining techniques to detect people’s preference for further recommendation, such as movie reviews and restaurant reviews. In this assignment, we need to establish a sentiment classiﬁcation model for the given sentence. In this project I have implemented a CNN, RNN and RNF model for sentence sentiment classiﬁcation.

## Dataset preparation
Each dataset should be formed by 3 files: `train.txt`, `dev.txt`, `test.txt`, each having the following structure. Make sure the 3 files are placed the same folder.
```
  l1 sentence1
  l2 sentence2
  ...
  lN sentenceN
```
You can find some examples in the directories `dataset/sst5` and `dataset/sst2` taken from Stanford Sentiment Treebank datasbase with 5 and 2 classes respectively.

## Data preprocessing
During the preprocessing phase the following pipeline will be applied to each sentence in each set: 
- Remtion of punctuaction.
- Lower case conversion.
- Remotion of non-purely alphabetical words.
- Filtering of stop words.
- Remotion of words less than two characters long.
For more details about the preprocessign phase tou can refer to the `loader.py` file.

## Pretrained embedding
You can download a pretrained embadding such as Glove (https://nlp.stanford.edu/projects/glove/) to initialize your embedding matrix. Once downloaded, put it in the folder `embed` (ex: `embed/glove6B.300d.txt`).

## CNN (Convolutional network)
Train a convolutional network.
```
  python main.py -m "cnn" -dp "dataset/sst5"
```
Train a convolutional network with dropout and embedding dropout.
```
  python main.py -m "cnn" -dp "dataset/sst5" -dr 0.4 -ed 0.4
```
Change number, size and depth of its filters.
```
  python main.py -m "cnn" -dp "dataset/sst5" -fs "2,3,4,5" -nf 50
```
Train it with pretrained embedding.
```
  python main.py -m "cnn" -dp "dataset/sst5" -ep "embed/glove6B.300d.txt"
```
Load a pretrained model.
```
  python main.py -m "cnn" -dp "dataset/sst5" -lp "model/cnn/model.tar"
```
Train on a dataset with binary labels.
```
  python main.py -m "cnn" -dp "dataset/sst2" -nc 2
```
Finally, you can use the command `--help` to visualize the full list of parameters you can fine tune as well as their description.

## RNN (LSTM)
Train a recurrent network.
```
  python main.py -m "rnn" -dp "dataset/sst5"
```
Change the number of layers and hidden size.
```
  python main.py -m "rnn" -dp "dataset/sst5" -nl 2 -hs 100
```

## RNF (Recurrent neural filter)
Train a convolutional network with recurrent neural filter.
```
  python main.py -m "rnf" -dp "dataset/sst5"
```
Change the size of the recurrent filter
```
  python main.py -m "rnf" -dp "dataset/sst5" -rnfs 10
```

## References

- https://github.com/bentrevett/pytorch-sentiment-analysis
- https://github.com/avinashsai/Recurrent-Neural-Filters
- https://github.com/bloomberg/cnn-rnf
