

class Voc:
    def __init__(self):
        self.trimmed = None
        self.word2index = {}  # mapping from words to indexes
        self.word2count = {}  # count of each word
        self.PAD_token = 0  # Used for padding short sentences
        self.index2word = {self.PAD_token: "PAD"}  # reverse mapping of indexes to words
        self.num_words = 1  # Count SOS, EOS, PAD

    def addSentences(self, sentences):
        """Add all words in a list sentence to the vocabulary"""
        for line in sentences:
            self.addSentence(line)

    def addSentence(self, sentence):
        """Add all words of a sentence to the vocabulary"""
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        """Add one word to the vocabulary"""
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        """Remove words below a certain count threshold"""

        self.trimmed = min_count
        keep_words = []
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)
        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))
        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {self.PAD_token: "PAD"}
        self.num_words = 1  # Count default tokens
        for word in keep_words:
            self.addWord(word)

    def indexesFromSentence(self, sentence):
        """Convert words of a sentence to their indexes"""
        return [self.word2index[word] for word in sentence.split(' ')]

    def pad_sentences(self, sentences, max_length, pad_direction='right'):
        """Pad all sentences in a list of lines"""
        padded_sentences = []
        for line in sentences:
            padded_sentences.append(self.pad_sentence(line, max_length, pad_direction))
        return padded_sentences

    def pad_sentence(self, sentence, max_length, pad_direction='right'):
        """Pad a sentence on the left or on the right"""
        indices = self.indexesFromSentence(sentence)
        if max_length - len(indices) > 0:
            padding = [self.PAD_token] * (max_length - len(indices))
            if pad_direction == 'right':
                return indices + padding
            if pad_direction == 'left':
                return padding + indices
        return indices[:max_length]

    def get_words(self):
        """Get all words in the vocabulary (includes PAD)"""
        return self.word2index.keys()

    def fix_sentences(self, sentences):
        """Remove words not present in the vocabulary from a list of sentences"""
        fixed_sentences = []
        for line in sentences:
            fixed_sentences.append(self.fix_sentence(line))
        return fixed_sentences

    def fix_sentence(self, sentence):
        """Remove words not present in the vocabulary from a sentence"""
        words = sentence.split()
        words = [w for w in words if w in self.get_words()]
        fixed_sentence = " ".join(words)
        return fixed_sentence
