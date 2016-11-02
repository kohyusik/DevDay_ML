import tensorflow as tf
import numpy as np

xy = np.loadtxt('target.txt', unpack=True, dtype='float32', delimiter=',')
x_data = np.transpose(xy[0:-1, 0:10])
y_data = np.transpose(xy[-1, 0:10])
#mat_modul = tf.constant([[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1]])

mat_modul = tf.constant([[1.,0.,0.,0.,0.], [0.,4.,0.,0.,0.], [0.,0.,1.,0.,0.], [0.,0.,0.,0.5,0.], [0.,0.,0.,0.,25.]])


sess = tf.Session()
print 'x : \n' , x_data
print 'mat_modul  : \n' , sess.run(mat_modul)

h = tf.matmul(x_data, mat_modul)



print sess.run(h)
print 'x : \n' , x_data
for step in xrange(5):
	data = x_data[-1 - step, :]
	print data[0], ' ******** ',data[2]

if 1 == 2 :
	print 'true'
else :
	print 'false'
	

# 20000 0.463418 [[ -1.62390459e+00   5.51483943e-04   3.31994928e-02  -1.02921817e-02 -1.23588722e-02]]

