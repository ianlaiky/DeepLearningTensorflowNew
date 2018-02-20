import tensorflow as tf

# step 1: pre-process the data


from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnist", one_hot=True)

X_train = mnist.train.images
y_train = mnist.train.labels
X_test = mnist.test.images
y_test = mnist.test.labels

L1 = 200
L2 = 100
L3 = 60
L4 = 30
L5 = 30
L6 = 20

x = tf.placeholder(tf.float32, [None, 784],name="X")
y = tf.placeholder(tf.float32, [None, 10])
W1 = tf.Variable(tf.truncated_normal([784, L1], stddev=0.1))
b1 = tf.Variable(tf.truncated_normal([L1], stddev=0.1))
W2 = tf.Variable(tf.truncated_normal([L1, L2], stddev=0.1))
b2 = tf.Variable(tf.truncated_normal([L2], stddev=0.1))
W3 = tf.Variable(tf.truncated_normal([L2, L3], stddev=0.1))
b3 = tf.Variable(tf.truncated_normal([L3], stddev=0.1))
W4 = tf.Variable(tf.truncated_normal([L3, L4], stddev=0.1))
b4 = tf.Variable(tf.truncated_normal([L4], stddev=0.1))
W5 = tf.Variable(tf.truncated_normal([L4, L5], stddev=0.1))
b5 = tf.Variable(tf.truncated_normal([L5], stddev=0.1))
W6 = tf.Variable(tf.truncated_normal([L5, L6], stddev=0.1))
b6 = tf.Variable(tf.truncated_normal([L6], stddev=0.1))

W7 = tf.Variable(tf.truncated_normal([L6, 10], stddev=0.1))
b7 = tf.Variable(tf.truncated_normal([10], stddev=0.1))

# step 2: setup the model

Y1 = tf.nn.relu(tf.matmul(x, W1) + b1)
Y2 = tf.nn.relu(tf.matmul(Y1, W2) + b2)
Y3 = tf.nn.relu(tf.matmul(Y2, W3) + b3)
Y4 = tf.nn.relu(tf.matmul(Y3, W4) + b4)
Y5 = tf.nn.relu(tf.matmul(Y4, W5) + b5)
Y6 = tf.nn.relu(tf.matmul(Y5, W6) + b6)


Ylogits = tf.matmul(Y6,W7)+b7
yhat = tf.nn.softmax(Ylogits,name="yhat")

# step 3: define the loss function

# loss = -tf.reduce_sum(y * tf.log(yhat))


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Ylogits,labels=y))

# step 4: define the optimiser


train = tf.train.GradientDescentOptimizer(0.55).minimize(loss)

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





saver = tf.train.Saver()
saver.save(sess,"./models/mnist2/mnist2.cpkt")
