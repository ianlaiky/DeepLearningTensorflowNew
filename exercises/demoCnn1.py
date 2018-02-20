#step 1: load the data

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist", one_hot=True,reshape=False,validation_size=0)# reshape is to keep the original shape instead of 784




X = tf.placeholder(tf.float32,[None,28,28,1])
y = tf.placeholder(tf.float32,[None,10])

L1 = 16# 1st conv layer filter
L2 = 32# 2nd conv layer filter
L3 = 64# fully conencted layer


W1 = tf.Variable(tf.truncated_normal([3,3,1,L1],stddev=0.1)) #filter size (3 by 3), Number of image (1) , number of filter (L1)
B1 = tf.Variable(tf.truncated_normal([L1],stddev=0.1))
W2 = tf.Variable(tf.truncated_normal([3,3,L1,L2],stddev=0.1))
B2 = tf.Variable(tf.truncated_normal([L2],stddev=0.1))
W3 = tf.Variable(tf.truncated_normal([7*7*L2,L3],stddev=0.1))
B3 = tf.Variable(tf.truncated_normal([L3],stddev=0.1))
W4 = tf.Variable(tf.truncated_normal([L3,10],stddev=0.1))
B4 = tf.Variable(tf.truncated_normal([10],stddev=0.1))


#step 2 setup the model

Y1 = tf.nn.relu(tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding='SAME')+B1)
Y1 = tf.nn.max_pool(Y1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
Y2 = tf.nn.relu(tf.nn.conv2d(Y1,W2,strides=[1,1,1,1],padding='SAME')+B2)
Y2 = tf.nn.max_pool(Y2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


#step 3 Flatten

YY = tf.reshape(Y2,shape=[-1,7*7*L2])
Y3 = tf.nn.relu(tf.matmul(YY,W3)+B3)

Ylogits = tf.matmul(Y3,W4)+B4
yhat = tf.nn.softmax(Ylogits)



#step 4 Loss function

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Ylogits, labels=y))




#step 5
train = tf.train.AdamOptimizer(0.01).minimize(loss)


# accuracy of the trained model
is_correct = tf.equal(tf.argmax(y,1),tf.argmax(yhat,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

