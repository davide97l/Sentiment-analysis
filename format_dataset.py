import os


def normalize(line):
    """Retrieve sentence text and label from tree structure"""
    line = line.replace("(", "").replace(")", "")
    line = line.split()
    label = line[0]
    sentence = [w for w in line if w not in ['0', '1', '2', '3', '4']]
    sentence = " ".join(sentence)
    return label, sentence


def read_file(filename):
    """Read a file line by line and normalize them"""
    lines = []
    f = open(filename, "r")
    for l in f:
        l = " ".join(normalize(l)) + "\n"
        lines.append(l)
    f.close()
    return lines


if __name__ == '__main__':
    """Normalize file train.txt, dev.txt and test.txt"""

    dataset_path = os.path.join("trainDevTestTrees_PTB", "trees")
    clean_dataset_path = "dataset"

    if not os.path.exists(clean_dataset_path):
        os.makedirs(clean_dataset_path)

    train = "train.txt"
    dev = "dev.txt"
    test = "test.txt"
    
    dev_file = read_file(os.path.join(dataset_path, dev))
    train_file = read_file(os.path.join(dataset_path, train))
    test_file = read_file(os.path.join(dataset_path, test))

    f = open(os.path.join(clean_dataset_path, "dev.txt"), "w")
    f.writelines(dev_file)
    f.close()
    f = open(os.path.join(clean_dataset_path, "train.txt"), "w")
    f.writelines(train_file)
    f.close()
    f = open(os.path.join(clean_dataset_path, "test.txt"), "w")
    f.writelines(test_file)
    f.close()
