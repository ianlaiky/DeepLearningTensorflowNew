import tensorflow as tf

# step 1: pre-process the data


from tensorflow.examples.tutorials.mnist import input_data
#skipped
# CIFAR-10 dataset from Keras
from tensorflow.python.keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = cifar10.train.images
y_train = cifar10.train.labels
X_test = cifar10.test.images
y_test = cifar10.test.labels

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.truncated_normal([784, 10], stddev=0.1))
b = tf.Variable(tf.truncated_normal([10], stddev=0.1))

# step 2: setup the model

yhat = tf.nn.softmax(tf.matmul(x, W) + b)

# step 3: define the loss function

loss = -tf.reduce_sum(y * tf.log(yhat))

# step 4: define the optimiser


train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# step 5: train the mode

is_correct = tf.equal(tf.argmax(y, 1), tf.argmax(yhat, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    batch_X, batch_y = mnist.train.next_batch(100)
    train_data = {x: batch_X, y: batch_y}
    sess.run(train, feed_dict=train_data)
    print(i + 1, "Training Accuracy = ", sess.run(accuracy, feed_dict=train_data))

# step 6:evaulate the model

test_data = {x: X_test, y: y_test}
print("Testing accuracy =", sess.run(accuracy, feed_dict=test_data))

# step 7: save the model
