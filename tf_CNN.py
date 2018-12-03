import tensorflow as tf
from numpy.random import RandomState
batch_size = 10
# 定义权重
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
# 定义输入输出，使用none先占位
x = tf.placeholder(tf.float32, [None, 2])
y = tf.placeholder(tf.float32, [None, 1])
# 使用relu函数进行激活
a = tf.nn.relu(tf.matmul(x, w1))
yhat = tf.nn.relu(tf.matmul(a, w2))
# 定义交叉熵函数
cross_entropy = -tf.reduce_mean(
    y * tf.log(tf.clip_by_value(yhat,1e-10,1.0)))
train_step = tf.train.AdamOptimizer(
    0.001).minimize(cross_entropy)
rdm = RandomState(1)
data_size = 512
X = rdm.rand(data_size, 2)
Y = [[int(x1+x2 < 1)] for (x1, x2) in X]
# 开始训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(w1))
    print(sess.run(w2))
    steps = 11000
    for i in range(steps):
        start = i*batch_size % data_size
        end = min(start+batch_size, data_size)
        sess.run(train_step, feed_dict={x: X[start: end],
                                        y: Y[start: end]})
        if i % 1000 == 0:
            training_loss=sess.run(cross_entropy,
                                   feed_dict={x:X,y:Y})
            print("After %d iterations, the loss is %g"
                  % (i, training_loss))
