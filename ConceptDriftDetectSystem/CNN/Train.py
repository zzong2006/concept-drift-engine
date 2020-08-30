#############################################################################
# 이하 신경망 학습 수행부
#############################################################################
import tensorflow as tf
from CNN import CNN_TF as CNN
from CNN import Input_train as Input
import os
from datetime import datetime

SAVE_DIR = "./trained_model" + "/"

def train(SAVE_DIR, TRAIN_NUM, KEEP_PROB):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:  
        saver = tf.train.Saver()                                # 학습이 끝난 모델 저장을 위한 saver 객체

        ckpt = tf.train.get_checkpoint_state(SAVE_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            print("checkpoint 파일이 이미 존재합니다. 삭제하고 진행하겠습니다.")
            tf.gfile.DeleteRecursively(SAVE_DIR)
            # tf.gfile.MakeDirs(SAVE_DIR)
            sess.run(tf.global_variables_initializer())
            #saver.restore(sess, SAVE_DIR)
        else:
            print("checkpoint 파일이 존재하지 않습니다. 새로 생성합니다.")
            sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()                          # 기본 큐 코디네이터 생성. 스레드들을 관리 가능
        threads = tf.train.start_queue_runners(coord=coord)     # 큐 러너 생성.

        # 데이터를 모으는 파트부. CSV를 한 줄씩 읽어내, batch 크기만큼의 데이터 덩이를 만들어낸다.
        # for j in range(NUM_OF_BATCH):
        
        ''' 위의 텐서플로우에서 다 그래프로 구성했으므로 이제 필요없음
        # 레이블 벡터 생성
        labelv = numpy.zeros(10) + 0.01
        labelv[int(example[0])] = 0.99

        # 값 벡터 생성
        scaled_values = (numpy.asfarray(example[1:]) / 255.0 * 0.99) + 0.01
        if j == 0:
            batch = [[scaled_values],[labelv]]    
        else:
            batch[0].append(scaled_values)
            batch[1].append(labelv)
        '''

        for i in range(TRAIN_NUM+1):
            batch_labels, batch_values = sess.run([Input.labels, Input.values])
            if (i % 10 == 0):
                train_accuracy = sess.run(CNN.accuracy, feed_dict={
                    CNN.x: batch_values, CNN.y_:batch_labels, CNN.keep_prob: 1.0})
                print("%s: step %d, training accuracy %g"% (datetime.now(), i, train_accuracy) )
            sess.run(CNN.train_step,feed_dict={CNN.x: batch_values, CNN.y_:batch_labels, CNN.keep_prob: KEEP_PROB})

        '''
        for i in range(100):
            examples = sess.run(nums)
            
            print(examples)
        '''
        coord.request_stop()
        coord.join(threads)          

        if tf.gfile.Exists(SAVE_DIR):
            tf.gfile.DeleteRecursively(SAVE_DIR)
        tf.gfile.MakeDirs(SAVE_DIR)
        save_path = saver.save(sess, SAVE_DIR)
        print("Model saved in file : %s"%(save_path))