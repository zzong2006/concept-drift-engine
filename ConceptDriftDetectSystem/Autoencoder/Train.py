import tensorflow as tf
from Autoencoder import Autoencoder_TF as AE, Input_train as Input
import configparser
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# import 

# Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data", one_hot=True, validation_size = 100)


# Parameters
display_step = 10
examples_to_show = 12

################################################
# csv 설정값 읽어오는 코드
################################################
CONF_FILE = "options.ini"
config = configparser.ConfigParser()
config.read(CONF_FILE)

# 전역 설정값 로드
section = "GENERAL"
DATA_FOLDER = "./" + config.get(section, 'DATA FOLDER NAME')     # 데이터 폴더는 공통으로 사용할 것

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

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_FOLDER)
        if ckpt and ckpt.model_checkpoint_path:
            print("checkpoint 파일이 이미 존재합니다. 삭제하고 진행하겠습니다.")
            tf.gfile.DeleteRecursively(MODEL_SAVE_FOLDER)
            sess.run(tf.global_variables_initializer())
            #saver.restore(sess, SAVE_DIR)
        else:
            print("checkpoint 파일이 존재하지 않습니다. 새로 생성합니다.")
            sess.run(tf.global_variables_initializer())             # 

        coord = tf.train.Coordinator()                          # 기본 큐 코디네이터 생성. 스레드들을 관리 가능
        threads = tf.train.start_queue_runners(sess = sess, coord=coord)     # 큐 러너 생성.

        # sess.run(init)

        # Training cycle
        # 1회 배치를 꺼냄
        for i in range(1, train_num + 1):
            batch_values = sess.run(Input.values)

            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([AE.optimizer, AE.cost], feed_dict={AE.X: batch_values})
            # Display logs per epoch step
            if i % display_step == 0 or i == 1:
                print("%s step: %04d," % (datetime.now(), i), "cost=", "{:.9f}".format(c))

        coord.request_stop()
        coord.join(threads)

        print("Optimization Finished! Model saving...")

        if tf.gfile.Exists(MODEL_SAVE_FOLDER):
            tf.gfile.DeleteRecursively(MODEL_SAVE_FOLDER)
        tf.gfile.MakeDirs(MODEL_SAVE_FOLDER)
        save_path = saver.save(sess, MODEL_SAVE_FOLDER)
        print("Model saved in file : %s"%(save_path))
         
    pass

#######################################################################################################
# 단독으로 실행될 일이 없을 것 같음
if __name__ == "__main__":
    # split_data_file_to_file(INPUT_FILE_NAME, OUTPUT_FILE_NAME, WINDOW_SIZE, SLIDE_STEP_SIZE)
    pass