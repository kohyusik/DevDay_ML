import tensorflow as tf
import numpy as np

xy = np.loadtxt('target.txt', unpack=True, dtype='float32', delimiter=',')
x_data = np.transpose(xy[0:-1]) # 117600
y_data = xy[-1]


mat_modul = tf.constant([[1.,0.,0.,0.,0.], [0.,4.,0.,0.,0.], [0.,0.,1.,0.,0.], [0.,0.,0.,0.5,0.], [0.,0.,0.,0.,25.]])
modulation = tf.matmul(x_data, mat_modul)

sess2 = tf.Session()
sess2.run(mat_modul)
x_data = np.transpose(sess2.run(modulation))


#print 'mat_modul  : \n' , mat_modul
#print 'xy : \n' , xy
print 'x  : \n' , x_data
print 'y  : \n' , y_data
#print 'size : ' , len(x_data)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Try to find value for W and b that compute y_data = W * x_data + b
# (We know that W should be 1 and b 0, but TensorFlow will
# figure that out for us.)

W = tf.Variable(tf.random_uniform([1, len(x_data)], -0.5, 0.5))
#W = tf.constant([[0.82699066, -0.04147232, -0.02248367, -0.07687526, -0.02953624]])

# Our hypothesis
h = tf.matmul(W, X)
hypothesis = tf.div(1., 1.+tf.exp(-h))
# cost function
cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))

# Minimize
a = tf.Variable(0.01) # Learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# Before starting, initialize the variable. We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# Fit the line.
for step in xrange(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 20 == 0:
        print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W)
# W = [[-0.98341316  0.35061496 -0.81121802  0.32535654 -0.88001603]]
# W = [[ 0.82699066 -0.04147232 -0.02248367 -0.07687526 -0.02953624]]
print '-----------------------------------------------'
# study hour attendance

# 
print sess.run(hypothesis, feed_dict={X:[[1], [20], [15], [5], [1]]})  # > 0.5

print sess.run(hypothesis, feed_dict={X:[[1,1,1,1,1], [4,8,12,16,20], [8,9,10,11,12], [5,10,15,20,25], [1,0,1,0,1]]})  # > 0.5

for step in xrange(100):
	data = x_data[:, -1 - step]
	result = y_data[-1 -step]
	print 'predict : ', sess.run(hypothesis, feed_dict={X:[[data[0]], [data[1]], [data[2]], [data[3]], [data[4]]]})# > 0.08567652
	print 'real : ', result

while 1 :
	day = int(raw_input("Enter day(0 ~ 6) ? \n")) * 4
	time = int(raw_input("Enter time(0 ~ 23) ? \n"))
	ages = int(raw_input("Enter ages(10 ~ 50) ? \n"))/2
	gender = int(raw_input("Enter gender(0 , 1) ? \n")) * 25
	print 'Open? ', sess.run(hypothesis, feed_dict={X:[[1], [day], [time], [ages], [gender]]}) # > 0.5


# 0.17135304 / 2 =  0.08567652





