import tensorflow as tf
import Autoencoder_TF as AE, Input_train as Input
import configparser
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# import 

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./sample/MNIST_data", one_hot=True, validation_size=100)

# Parameters
display_step = 10
examples_to_show = 12

################################################
# csv 설정값 읽어오는 코드
################################################
CONF_FILE = "../options.ini"
config = configparser.ConfigParser()
config.read(CONF_FILE)

# 전역 설정값 로드
section = "GENERAL"
DATA_FOLDER = "./" + config.get(section, 'DATA FOLDER NAME')  # 데이터 폴더는 공통으로 사용할 것
print(DATA_FOLDER)
section = "STREAM DATA"
MODEL_SAVE_FOLDER = "./" + config.get(section, 'WINDOW SIZE') + "/"

section = "DATA REDUCING"
MODEL_SAVE_FOLDER = "./" + config.get(section, 'MODEL SAVE FOLDER NAME') + "/"
train_num = int(config.get(section, 'TRAIN NUM'))
BATCH_SIZE = int(config.get(section, 'BATCH SIZE'))


################################################
# 이하 신경망 학습 코드
################################################
def train():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Launch the graph
    with tf.Session(config=config) as sess:
        # Model Saver op
        saver = tf.train.Saver()

        # ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_FOLDER)
        # if ckpt and ckpt.model_checkpoint_path:
        #     print("checkpoint 파일이 존재합니다. load 하겠습니다.")
        #     saver.restore(sess, MODEL_SAVE_FOLDER)
        # else:
        #     print("checkpoint 파일이 존재하지 않습니다. 프로그램을 종료합니다.")
        #     exit()

        coord = tf.train.Coordinator()  # 기본 큐 코디네이터 생성. 스레드들을 관리 가능
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # 큐 러너 생성.

        # Training cycle
        # 1회 배치를 꺼냄
        for i in range(1, train_num + 1):
            # batch_values = sess.run(Input.values)

            batch_values, _ = mnist.train.next_batch(BATCH_SIZE)

            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([AE.optimizer, AE.cost], feed_dict={AE.X: batch_values})
            # Display logs per epoch step
            if i % display_step == 0 or i == 1:
                print("%s step: %04d," % (datetime.now(), i), "cost=", "{:.9f}".format(c))

        print("Optimization Finished!")

        # Testing
        # Encode and decode images from test set and visualize their reconstruction.
        n = 4
        canvas_orig = np.empty((20 * n, 50 * n))
        canvas_recon = np.empty((20 * n, 50 * n))
        for i in range(n):
            # MNIST test set
            batch_values = sess.run(Input.values)
            # Encode and decode the digit image
            g = sess.run(AE.decoder_op, feed_dict={AE.X: batch_values})

            # Display original images
            for j in range(n):
                # Draw the original digits
                canvas_orig[i * 20:(i + 1) * 20, j * 50:(j + 1) * 50] = \
                    batch_values[j].reshape([20, 50])
            # Display reconstructed images
            for j in range(n):
                # Draw the reconstructed digits
                canvas_recon[i * 20:(i + 1) * 20, j * 50:(j + 1) * 50] = \
                    g[j].reshape([20, 50])

        print("Original Images")
        plt.figure(figsize=(n, n))
        plt.imshow(canvas_orig, origin="upper", cmap="gray")
        # plt.imshow(canvas_orig, origin="upper")
        plt.show()

        print("Reconstructed Images")
        plt.figure(figsize=(n, n))
        plt.imshow(canvas_recon, origin="upper", cmap="gray")
        # plt.imshow(canvas_recon, origin="upper")
        plt.show()

        coord.request_stop()
        coord.join(threads)


#######################################################################################################
# 단독으로 실행될 일이 없을 것 같음
if __name__ == "__main__":
    # split_data_file_to_file(INPUT_FILE_NAME, OUTPUT_FILE_NAME, WINDOW_SIZE, SLIDE_STEP_SIZE)
    pass
