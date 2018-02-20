import tensorflow as tf

print(tf.__version__)

# step 1: create a graph

# a = tf.constant(2.5,dtype=tf.float32)
# b = tf.constant(10,dtype=tf.float32)
# c = tf.multiply(a, b)



# a = tf.constant(3,dtype=tf.float32)
# b = tf.constant(4.1,dtype=tf.float32)
# c = tf.constant(5,dtype=tf.float32)
# d = tf.multiply(a,b)
# e = tf.add(d,c)


# a = tf.constant([
#     [1,2],
#     [3,4]
#
# ])
#
#
# b = tf.constant([
#     [4,3],
#     [2,1]
#
# ])
#
# c = tf.matmul(a,b)
# d = tf.reduce_sum(a,axis=1)
# e = tf.argmax(b,axis=1)



# a = tf.zeros([2,3])




# a = tf.constant([
#     [1.1,1.2]
# ],dtype=tf.float32)
#
# b = tf.constant([
#     [1,2],
#     [3,4]
# ],dtype=tf.float32)
#
# c = tf.constant([
#     [2.3,2.5]
# ],dtype=tf.float32)
#
# y = tf.matmul(a,b)
# y = y+c

#
# a = tf.random_normal([2,3])
# b = tf.truncated_normal([2,3])


# a= tf.placeholder(tf.float32)
# b = tf.placeholder(tf.float32)
#
# c = tf.multiply(a,b)




# x = tf.placeholder(tf.float32)
# w = tf.placeholder(tf.float32)
# b = tf.placeholder(tf.float32)
#
# y = tf.add(tf.matmul(x,w),b)



#step 2: run the session
sess = tf.Session()

# print(sess.run(y,feed_dict={x:[[1.1,1.2]],w:[[1,2],[3,4]],b:[[2.3,2.5]]}))

