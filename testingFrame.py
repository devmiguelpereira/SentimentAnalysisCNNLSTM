# installing  certain packages and libraries

# pip install mxnet
# pip install mxnet-cu90

# Importing libraries
import os
import mxnet as mx

# Load categories
categories = ['Excellent', 'Very_good', 'Good', 'Average', 'Poor']

# Load the data in memory
MAX_ITEMS_PER_CATEGORY = 5000
