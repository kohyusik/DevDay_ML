import tensorflow as tf
import numpy as np

# kohyusik
# data file : target.txt (rows=117600)
xy = np.loadtxt('target.txt', unpack=True, dtype='float32', delimiter=',')
x_data = np.transpose(xy[1:-1]) # N x 4 Maxrix
y_data = np.reshape(xy[-1], (len(x_data), 1)) # N x 1 Maxrix
days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

# input data level tune
mat_modul = tf.constant([[4.,0.,0.,0.], [0.,1.,0.,0.], [0.,0.,0.5,0.], [0.,0.,0.,25.]])
modulation = tf.matmul(x_data, mat_modul)

sess2 = tf.Session()
sess2.run(mat_modul)
x_data = sess2.run(modulation)

print 'x  : \n' , x_data
print 'y  : \n' , y_data


X = tf.placeholder(tf.float32, name='x-input')
Y = tf.placeholder(tf.float32, name='y-output')


w1 = tf.Variable(tf.random_uniform([4, 5], -0.5, 0.5), name='weight1')
w2 = tf.Variable(tf.random_uniform([5, 10], -0.5, 0.5), name='weight2')
w3 = tf.Variable(tf.random_uniform([10, 10], -0.5, 0.5), name='weight3')
w4 = tf.Variable(tf.random_uniform([10, 10], -0.5, 0.5), name='weight4')
w5 = tf.Variable(tf.random_uniform([10, 10], -0.5, 0.5), name='weight5')
w6 = tf.Variable(tf.random_uniform([10, 10], -0.5, 0.5), name='weight6')
w7 = tf.Variable(tf.random_uniform([10, 10], -0.5, 0.5), name='weight7')
w8 = tf.Variable(tf.random_uniform([10, 1], -0.5, 0.5), name='weight8')

b1 = tf.Variable(tf.zeros([5]), name="Bias1")
b3 = tf.Variable(tf.zeros([10]), name="Bias3")
b2 = tf.Variable(tf.zeros([10]), name="Bias2")
b4 = tf.Variable(tf.zeros([10]), name="Bias4")
b5 = tf.Variable(tf.zeros([10]), name="Bias5")
b6 = tf.Variable(tf.zeros([10]), name="Bias6")
b7 = tf.Variable(tf.zeros([10]), name="Bias7")
b8 = tf.Variable(tf.zeros([1]), name="Bias8")

L2 = tf.nn.relu(tf.matmul(X, w1) + b1)
L3 = tf.nn.relu(tf.matmul(L2, w2) + b2)
L4 = tf.nn.relu(tf.matmul(L3, w3) + b3)
L5 = tf.nn.relu(tf.matmul(L4, w4) + b4)
L6 = tf.nn.relu(tf.matmul(L5, w5) + b5)
L7 = tf.nn.relu(tf.matmul(L6, w6) + b6)
L8 = tf.nn.relu(tf.matmul(L7, w7) + b7)

# hypothesis
hypothesis = tf.sigmoid(tf.matmul(L8, w8) + b8)
#h = tf.matmul(L8, w8) + b8
#hypothesis = tf.div(1., 1.+tf.exp(-h))

# cost function
with tf.name_scope('cost') as scope:
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

with tf.name_scope('train') as scope:
    a = tf.Variable(0.1)
    optimizer = tf.train.GradientDescentOptimizer(a)
    train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    for step in xrange(5000):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 50 == 0:
            print step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(w1), sess.run(w2)

    correct_prediction = tf.equal(tf.floor(hypothesis + 0.5), Y)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print sess.run([hypothesis, tf.floor(hypothesis + 0.5), correct_prediction], feed_dict={X: x_data, Y: y_data})
    print "accuracy", accuracy.eval({X: x_data, Y: y_data})

    print "TEST INPUT : \n"
    print sess.run(hypothesis, feed_dict={X: [[12, 13, 50/2, 0*25]]}) > 0.5
    print sess.run(hypothesis, feed_dict={X: [[20, 10, 25, 0]]}) > 0.5
    # print sess.run(hypothesis, feed_dict={X:[[20., 15., 5., 1.], [20., 15., 5., 1.]] })  # > 0.5
    # print sess.run(hypothesis, feed_dict={X:[[20, 15, 5, 1]] })[0][0]  # > 0.5

    #input
    while 1 :
        ages = int(raw_input("Enter ages(10 ~ 50) ? \n")) / 2.
        gender = int(raw_input("Enter gender(0 , 1) ? \n")) * 25.

        print '######## RESULT ########'
        for day in xrange(7) :
            print '###', days[day], 'open rate ###'
            day = day * 4
            for time in xrange(24) :
                result = sess.run(hypothesis, feed_dict={X:[[day, time, ages, gender]]})[0][0]
                print '-',time,'\t:',round(result,4) > 0.5



