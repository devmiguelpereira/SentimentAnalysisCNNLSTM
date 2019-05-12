# installing  certain packages and libraries

# pip install mxnet
# pip install mxnet-cu90

# Importing libraries
import shutil,os
import pandas as pd
import gzip
import mxnet as mx

# creating variables
folder = 'Data'
prefix = 'reviews_'
suffix = '.json.gz'


# Load Data
categories = ['Excellent', 'Very_good', 'Good', 'Average', 'Poor']

os.chdir('\')

# Load the data in memory
MAX_ITEMS_PER_CATEGORY = 2469


# Helper functions to read from the .json.gzip files
def parse(path):
    g = gzip.open(path, 'rb')
    for line in g:
        yield eval(line)


def get_dataframe(path, num_lines):
    i = 0
    df = {}
    for d in parse(path):
        if i > num_lines:
            break
        df[i] = d
        i += 1

    return pd.DataFrame.from_dict(df, orient='index')


# Loading data from file if exist
try:
    data = pd.read_pickle('pickleddata.pkl')
except:
    data = None


if data is None:
    data = pd.DataFrame(data={'X':[],'Y':[]})
    for index, category in enumerate(categories):
        df = get_dataframe("{}/{}{}{}".format(folder, prefix, category, suffix), MAX_ITEMS_PER_CATEGORY)
        # Each review's summary is prepended to the main review text
        df = pd.DataFrame(data={'X':(df['summary']+' | '+df['reviewText'])[:MAX_ITEMS_PER_CATEGORY],'Y':index})
        data = data.append(df)
        print('{}:{} reviews'.format(category, len(df)))

    # Shuffle the samples
    data = data.sample(frac=1)
    data.reset_index(drop=True, inplace=True)
    # Saving the data in a pickled file
    pd.to_pickle(data, 'pickleddata.pkl')

print('End of the line')