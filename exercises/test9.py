
import tensorflow as tf

a = tf.constant(3,name="a",dtype=tf.float32)
b = tf.constant(4,name="b",dtype=tf.float32)

with tf.name_scope('multiply'):
    c = tf.multiply(a,b,name="c")
with tf.name_scope('div'):
    d = tf.div(a,b,name="c")



sess = tf.Session()

tf.summary.scalar('c',c)
tf.summary.scalar('d',d)
merged_summary = tf.summary.merge_all()




#visualise
#Tensorboard --logdir demo9
writer = tf.summary.FileWriter('./tb/demo9')
writer.add_graph(sess.graph)
writer.add_summary(sess.run(merged_summary))
print(sess.run(c))
print(sess.run(d))
