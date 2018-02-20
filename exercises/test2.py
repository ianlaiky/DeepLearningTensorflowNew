import tensorflow as tf


# step 1 : preprocess data

x_train = [1., 2., 3., 4., 5.]
y_train = [0, -1.5, -2.6, -4.3, -4.8]

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
w = tf.Variable([0.1],tf.float32)
b = tf.Variable([0.1],tf.float32)


#  import matplotlib.pyplot as plt
# plt.scatter(x_train,y_train)
# plt.show()


# step 2: define the mode

yhat = tf.multiply(x,w)+b



# step 3: define loss function

loss = tf.reduce_sum(tf.square(yhat-y))

# step 4: define the optimizer

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# step 5: train the model

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    sess.run(train,{x:x_train,y:y_train})

# step 6 : evaluate the mode

import matplotlib.pyplot as plt
plt.scatter(x_train,y_train)
yhat = sess.run(tf.multiply(x_train,w)+b)
plt.plot(x_train,yhat,'r')
plt.show()

# step 7: save the model

print("w = ",sess.run(w))
print("b = ",sess.run(b))