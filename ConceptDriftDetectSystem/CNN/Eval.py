#############################################################################
# 이하 신경망 평가 수행부
#############################################################################
import tensorflow as tf
from CNN import CNN_TF as CNN
from CNN import Input_eval as Input
import math
import os

SAVE_DIR = "./trained_model" + "/"

EVAL_DATA_QUANTITY = 192118  # 평가를 위한 데이터 수


def eval(SAVE_DIR, EVAL_DATA_QUANTITY):
    BATCH_QUANTITY_FOR_EVAL = math.ceil(EVAL_DATA_QUANTITY / Input.BATCH_SIZE)  # 평가를 위해 수행해야하는 배치 수

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()  # 학습이 끝난 모델 로드를 위한 saver 객체

        ckpt = tf.train.get_checkpoint_state(SAVE_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            print("checkpoint 파일이 존재합니다. load 하겠습니다.")
            saver.restore(sess, SAVE_DIR)
        else:
            print("checkpoint 파일이 존재하지 않습니다. 프로그램을 종료합니다.")
            exit()

        coord = tf.train.Coordinator()  # 기본 큐 코디네이터 생성. 스레드들을 관리 가능
        threads = tf.train.start_queue_runners(coord=coord)  # 큐 러너 생성.

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

        print("모델의 로드에 성공했습니다. 모든 학습데이터에 대하여 신경망의 적중률을 계산합니다.")

        correct_count_total = 0

        for i in range(BATCH_QUANTITY_FOR_EVAL):
            batch_labels, batch_values = sess.run([Input.labels, Input.values])  # 입력을 가져옴

            correct_count_per_batch = sess.run(CNN.correct_answer, feed_dict={
                CNN.x: batch_values, CNN.y_: batch_labels, CNN.keep_prob: 1.0})

            if i % 10 == 0:
                print("step %d, correct_count_per_batch %d" % (i, correct_count_per_batch))
                pass

            correct_count_total += correct_count_per_batch

        print("적중률: %.2f%%" % (correct_count_total / (BATCH_QUANTITY_FOR_EVAL * Input.BATCH_SIZE) * 100))  # 최종 정확도 출력
        '''
        for i in range(100):
            examples = sess.run(nums)
            
            print(examples)
        '''
        coord.request_stop()
        coord.join(threads)
