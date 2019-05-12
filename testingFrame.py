def main():

    # installing  certain packages and libraries

    # pip install mxnet
    # pip install mxnet-cu90

    # Importing libraries
    import shutil, os

    import pandas as pd
    import json
    import mxnet as mx

    # creating variables
    base_path = 'Dataset'
    folder = 'data'
    prefix = 'reviews_'
    suffix = '.csv'

    # Load Data
    categories = ['Excellent', 'Very_good', 'Good', 'Average', 'Poor']

    # Load the data in memory
    MAX_ITEMS_PER_CATEGORY = 25000

    # df = pd.read_csv('Average.csv', encoding='utf8')


    # Loading data from file if exist
    try:
        data = pd.read_pickle('pickleddata.pkl')
    except:
        data = None

    if data is None:
        data = pd.DataFrame(data={'X': [], 'Y': []})
        for index, category in enumerate(categories):
            df = pd.read_csv(category + suffix, encoding='utf8')
            df = pd.DataFrame(data={'X': (df['Review'])[:MAX_ITEMS_PER_CATEGORY], 'Y': index})
            data = data.append(df)
            print('{}:{} reviews'.format(category, len(df)))

        # Shuffle the samples
        data = data.sample(frac=1)
        data.reset_index(drop=True, inplace=True)
        # Saving the data in a pickled file
        pd.to_pickle(data, 'pickleddata.pkl')

    print('Value counts:\n', data['Y'].value_counts())
    for i, cat in enumerate(categories):
        print(i, cat)
    data.head()

    # Creating the dataset

    import multiprocessing
    from mxnet import nd, autograd, gluon
    from mxnet.gluon.data import ArrayDataset
    from mxnet.gluon.data import DataLoader
    import numpy as np

    ALPHABET = list(
        "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}")  # The 69 characters as specified in the paper
    ALPHABET_INDEX = {letter: index for index, letter in enumerate(ALPHABET)}  # { a: 0, b: 1, etc}
    FEATURE_LEN = 1014  # max-length in characters for one document
    NUM_WORKERS = 0  # number of workers used in the data loading
    BATCH_SIZE = 128  # number of documents per batch


    def encode(text):
        encoded = np.zeros([len(ALPHABET), FEATURE_LEN], dtype='float32')
        review = text.lower()[:FEATURE_LEN - 1:-1]
        i = 0
        for letter in text:
            if i >= FEATURE_LEN:
                break;
            if letter in ALPHABET_INDEX:
                encoded[ALPHABET_INDEX[letter]][i] = 1
            i += 1
        return encoded


    def transform(x, y):
        return encode(x), y

    split = 0.8
    split_index = int(split * len(data))
    train_data_X = data['X'][:split_index].as_matrix()
    train_data_Y = data['Y'][:split_index].as_matrix()
    test_data_X = data['X'][split_index:].as_matrix()
    test_data_Y = data['Y'][split_index:].as_matrix()
    train_dataset = ArrayDataset(train_data_X, train_data_Y).transform(transform)
    test_dataset = ArrayDataset(test_data_X, test_data_Y).transform(transform)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                                  last_batch='rollover')
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                                 last_batch='rollover')

    ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()

    NUM_FILTERS = 256  # number of convolutional filters per convolutional layer
    NUM_OUTPUTS = len(categories)  # number of classes
    FULLY_CONNECTED = 1024  # number of unit in the fully connected dense layer
    DROPOUT_RATE = 0.5  # probability of node drop out
    LEARNING_RATE = 0.001  # learning rate of the gradient
    MOMENTUM = 0.9  # momentum of the gradient
    WDECAY = 0.00001  # regularization term to limit size of weights

    net = gluon.nn.HybridSequential()
    with net.name_scope():
        net.add(gluon.nn.Conv1D(channels=NUM_FILTERS, kernel_size=7, activation='relu'))
        net.add(gluon.nn.MaxPool1D(pool_size=3, strides=3))
        net.add(gluon.nn.Conv1D(channels=NUM_FILTERS, kernel_size=7, activation='relu'))
        net.add(gluon.nn.MaxPool1D(pool_size=3, strides=3))
        net.add(gluon.nn.Conv1D(channels=NUM_FILTERS, kernel_size=3, activation='relu'))
        net.add(gluon.nn.Conv1D(channels=NUM_FILTERS, kernel_size=3, activation='relu'))
        net.add(gluon.nn.Conv1D(channels=NUM_FILTERS, kernel_size=3, activation='relu'))
        net.add(gluon.nn.Conv1D(channels=NUM_FILTERS, kernel_size=3, activation='relu'))
        net.add(gluon.nn.MaxPool1D(pool_size=3, strides=3))
        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.Dense(FULLY_CONNECTED, activation='relu'))
        net.add(gluon.nn.Dropout(DROPOUT_RATE))
        net.add(gluon.nn.Dense(FULLY_CONNECTED, activation='relu'))
        net.add(gluon.nn.Dropout(DROPOUT_RATE))
        net.add(gluon.nn.Dense(NUM_OUTPUTS))

if __name__ == "__main__":
    main()