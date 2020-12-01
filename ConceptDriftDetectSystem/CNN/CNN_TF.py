'''
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
'''

import tensorflow as tf
import configparser

################################################
# csv 설정값 읽어오는 코드
################################################
CONF_FILE = "options.ini"
config = configparser.ConfigParser()
config.read(CONF_FILE)

section = "STREAM DATA"
IMAGE_SIZE = int(config.get(section, 'WINDOW SIZE'))

section = "CLUSTERING"
LABEL_VECTOR_LENGTH = int(config.get(section, 'CLUSTER AMOUNT'))

# 전역 설정값 로드
section = "CNN"
NUM_OF_1ST_KERNEL = int(config.get(section, 'NUM OF 1ST KERNEL'))
NUM_OF_2ND_KERNEL = int(config.get(section, 'NUM OF 2ND KERNEL'))
L1_SIZE = int(config.get(section, 'L1 SIZE'))
L2_SIZE = int(config.get(section, 'L2 SIZE'))

#############################################################################
# 신경망
#############################################################################


x = tf.placeholder("float", shape=[None, IMAGE_SIZE])
y_ = tf.placeholder("float", shape=[None, LABEL_VECTOR_LENGTH])

x_image = tf.reshape(x, [-1, 1, IMAGE_SIZE, 1])


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')


# 1층
# NUM_OF_1ST_KERNEL = 128

W_conv1 = weight_variable([1, 5, 1, NUM_OF_1ST_KERNEL])
b_conv1 = bias_variable([NUM_OF_1ST_KERNEL])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

print(x_image)
print(h_conv1)
print(h_pool1)

# 2층
# NUM_OF_2ND_KERNEL = 96

W_conv2 = weight_variable([1, 5, NUM_OF_1ST_KERNEL, NUM_OF_2ND_KERNEL])
b_conv2 = bias_variable([NUM_OF_2ND_KERNEL])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # pool 레이어 1의 결과와
h_pool2 = max_pool_2x2(h_conv2)

print("Size of image after conv2: ", h_conv2.get_shape())
print("Size of image after pool2: ", h_pool2.get_shape())

IMAGE_SIZE_AT_FINAL = int(IMAGE_SIZE / (2 * 2))  # 폴링 레이어에 따라 분모가 달라짐. 1/2로 2회 줄어들은 것.

# 완전 연결 계층
# L1_SIZE = 1024

W_fc1 = weight_variable([1 * IMAGE_SIZE_AT_FINAL * NUM_OF_2ND_KERNEL, L1_SIZE])  # 192 was 7 * 7
b_fc1 = bias_variable([L1_SIZE])

h_pool2_flat = tf.reshape(h_pool2, [-1, IMAGE_SIZE_AT_FINAL * NUM_OF_2ND_KERNEL])  # 192 was 7 * 7
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

'''
# 계층 확장을 위해 추가중인 코드
L2_SIZE = 384
W_fc2 = weight_variable([1 * IMAGE_SIZE_AT_FINAL * NUM_OF_2ND_KERNEL, L2_SIZE])     # 차원을 어떻게 설정해야할까
b_fc2 = bias_variable([L2_SIZE])

h_fc1_flat = tf.reshape(h_fc1, [-1, IMAGE_SIZE_AT_FINAL * NUM_OF_2ND_KERNEL])       # 차원을 어떻게 설정해야할까(2)
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_flat, W_fc2) + b_fc2)
'''

# 드롭아웃 레이어
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 완전 연결 계층 2
# L2_SIZE = 512
W_fc2 = weight_variable([1 * L1_SIZE, L2_SIZE])
b_fc2 = bias_variable([L2_SIZE])

h_fc1_flat = tf.reshape(h_fc1_drop, [-1, L1_SIZE])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_flat, W_fc2) + b_fc2)

# 드롭아웃 레이어 2
# keep_prob = tf.placeholder("float") # 위에 이미 있으므로
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

# 소프트맥스 레이어
W_fc3 = weight_variable([L2_SIZE, LABEL_VECTOR_LENGTH])
b_fc3 = bias_variable([LABEL_VECTOR_LENGTH])

y_conv = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)  # 확률 벡터
top_k_op = tf.nn.top_k(y_conv, 1, False, None)

# 크로스 엔트로피, 평가를 위한 연산 정의
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))  # 정확히 맞춘 경우를 확인하려는 경우
# correct_prediction = tf.nn.in_top_k(y_conv, tf.argmax(y_,1), THRESHOLD_FOR_CDD)
# 벡터 내에서 신뢰도 순으로 k순위 안에 정답이 있는지 확인하려는 경우
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # 정확도를 %로 표현
correct_answer = tf.reduce_sum(tf.cast(correct_prediction, "int32"))  # 맞춘 갯수

'''
# session의 설정을 위한 코드 부분
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # GPU 메모리 사용률을 관찰하기 위한 코드

global_step = tf.contrib.framework.get_or_create_global_step()

sess = tf.Session(config = config)
sess.run(tf.global_variables_initializer())

'''

'''
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 1000 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={
                x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    sess.run(train_step,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g" % sess.run(
        h_fc1_drop, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
'''
