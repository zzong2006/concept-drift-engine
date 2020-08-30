import tensorflow as tf
from Autoencoder import Autoencoder_TF as AE, Input_reduce as Input
import configparser
import math
import csv

# import 

'''
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True, validation_size = 100)
'''

# Parameters
display_step = 1
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
training_epochs = 1             # 전체 데이터를 1회만 순회할 것임
BATCH_SIZE = int(config.get(section, 'BATCH SIZE'))

batch_per_epoch = 100   # 1 에폭을 학습하기 위하여 필요한 배치 수. 디폴트로 100 주었다

################################################
# 이하 신경망을 이용한 데이터 축소 코드
################################################
def datareduce(num_of_window_data, OUTPUT_DATA_PATH):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    batch_per_epoch = math.ceil(num_of_window_data / BATCH_SIZE)

    out_file = open(OUTPUT_DATA_PATH, 'w')
    out_file_csv = csv.writer(out_file, delimiter=',', lineterminator='\n')

    # out_file_temp = open("./TEMPDATA.csv", 'w')
    # out_file_temp_csv = csv.writer(out_file_temp, delimiter=',', lineterminator='\n')

    # Launch the graph
    with tf.Session(config=config) as sess:
        # Model Saver op
        saver = tf.train.Saver()

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_FOLDER)
        if ckpt and ckpt.model_checkpoint_path:
            print("checkpoint 파일이 존재합니다. load 하겠습니다.")
            saver.restore(sess, MODEL_SAVE_FOLDER)
        else:
            print("checkpoint 파일이 존재하지 않습니다. 프로그램을 종료합니다.")
            exit()

        coord = tf.train.Coordinator()                          # 기본 큐 코디네이터 생성. 스레드들을 관리 가능
        threads = tf.train.start_queue_runners(sess = sess, coord=coord)     # 큐 러너 생성.

        # sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            # batch_labels, batch_values = sess.run([Input.labels, Input.values])
    
            # Loop over all batches
            for i in range(batch_per_epoch):
                # 데이터를 불러옴
                batch_values = sess.run(Input.values)

                # Run optimization op (backprop) and cost op (to get loss value)
                encoded = sess.run(AE.encoder_op, feed_dict={AE.X: batch_values})

                # 생성된 데이터들을 순회하며 한 줄씩 써내려감
                for x in encoded:
                    out_file_csv.writerow(x)
                    pass
                pass
            pass

        coord.request_stop()
        coord.join(threads) 
        
        out_file.close()

        #out_file_temp.close()
    pass

#######################################################################################################
# 단독으로 실행될 일이 없을 것 같음
if __name__ == "__main__":
    # split_data_file_to_file(INPUT_FILE_NAME, OUTPUT_FILE_NAME, WINDOW_SIZE, SLIDE_STEP_SIZE)
    pass