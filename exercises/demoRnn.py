# step 1 load the data
from keras.datasets import imdb
import tensorflow as tf
from keras.preprocessing import sequence

import numpy as np
max_words = 2000
max_len = 80
rnn_size = 32

(X_train,y_train),(X_test,y_test)=imdb.load_data(num_words=max_words)

X_train = sequence.pad_sequences(X_train,maxlen=max_len,padding='pre',truncating='pre')

X_test = sequence.pad_sequences(X_test,maxlen=max_len,padding='pre',truncating='pre')
# print(X_train)

n_classes = len(np.unique(y_train))
y_train = np.eye(n_classes)[y_train]
y_test = np.eye(n_classes)[y_test]

X = tf.placeholder(tf.int32,[None,max_len])
y = tf.placeholder(tf.int32)
W = tf.Variable(tf.truncated_normal([rnn_size,n_classes],stddev=0.1))
B = tf.Variable(tf.truncated_normal([n_classes],stddev=0.1))
print(y_train)

# step 1 define the model
# step 1
# step 1
# step 1
# step 1