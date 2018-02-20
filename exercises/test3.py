import numpy as np
import tensorflow as tf
x_train = np.linspace(-10.0,10.0,20.0).astype(np.float32)
y_train = 2*x_train*x_train - 1 + 10.0*np.random.randn(len(x_train))



x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
w1 = tf.Variable([0.1],tf.float32)
w2 = tf.Variable([0.1],tf.float32)
b = tf.Variable([0.1],tf.float32)


# import matplotlib.pyplot as plt
# plt.scatter(x_train,y_train)
# plt.show()

yhat = w1*x_train*x_train+w2*x_train+b


loss = tf.reduce_sum(tf.square(yhat-y))


train = tf.train.GradientDescentOptimizer(0.000001).minimize(loss)


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    sess.run(train,{x:x_train,y:y_train})

# step 6 : evaluate the mode

import matplotlib.pyplot as plt
plt.scatter(x_train,y_train)
yhat = sess.run(w1*x_train*x_train+w2*x_train+b)
plt.plot(x_train,yhat,'r')
plt.show()

# step 7: save the model

print("w = ",sess.run(w1))
print("w = ",sess.run(w2))
print("b = ",sess.run(b))