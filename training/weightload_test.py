from mnist import input_data
import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

W_conv1 = weight_variable([3, 3, 1, 8])
b_conv1 = bias_variable([8])

x_image = tf.reshape(x, [-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

W_conv2 = weight_variable([3, 3, 8, 8])
b_conv2 = bias_variable([8])

h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([3, 3, 8, 16])
b_conv3 = bias_variable([16])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

W_fc1 = weight_variable([14 * 14 * 16, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_conv3, [-1, 14*14*16])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
for i in range(10000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("[float32] : ",W_fc2)
print("[float32 weight load] test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

W_conv1 = tf.cast(W_conv1, tf.float16)
b_conv1 = tf.cast(b_conv1, tf.float16)
W_conv2 = tf.cast(W_conv2, tf.float16)
b_conv2 = tf.cast(b_conv2, tf.float16)
W_conv3 = tf.cast(W_conv3, tf.float16)
b_conv3 = tf.cast(b_conv3, tf.float16)
W_fc1 = tf.cast(W_fc1, tf.float16)
b_fc1 = tf.cast(b_fc1, tf.float16)
W_fc2 = tf.cast(W_fc2, tf.float16)
b_fc2 = tf.cast(b_fc2, tf.float16)

print("[float16] : ",W_fc2)
print("[float16 weight load] test accuracy %g"%accuracy.eval(feed_dict={
                        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
