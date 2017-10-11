import tensorflow as tf
import numpy as np

x_data = np.float32(np.random.rand(1, 100))
y_data = np.dot([0.200], x_data) + 0.300

b = tf.Variable(tf.zeros([1]), name="v_b")
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0), name="v_w")
y = tf.matmul(W, x_data) + b

cost = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)

# with tf.name_scope("train"):
train = optimizer.minimize(cost)


init = tf.initialize_all_variables()
sess = tf.Session()

# create a log writer. run 'tensorboard --logdir=./logs/nn_logs'
writer = tf.train.SummaryWriter("./logs/sp2", sess.graph)  # for 0.8
merged = tf.merge_all_summaries()

sess.run(init)

for step in xrange(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print step, sess.run(W), sess.run(c)
