#############################################################################
# 이하 신경망 평가 수행부
#############################################################################
import tensorflow as tf
import CNN_TF as CNN
import Input_CDdetect as Input
import numpy as np
import math
import os

SAVE_DIR = "./trained_model" + "/"

EVAL_DATA_QUANTITY = 758  # 평가를 위한 데이터 수
BATCH_QUANTITY_FOR_EVAL = math.ceil(EVAL_DATA_QUANTITY / Input.NUM_OF_BATCH)  # 평가를 위해 수행해야하는 배치 수

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

    ###########################################################################
    # 이하 CD 검출부
    # 윈도우 A(이전 시점)와 윈도우 B(현재 시점)를 비교해가며 CD 여부를 검출한다
    ###########################################################################

    print(
        "CD 검출을 시작합니다.\n입력 윈도우의 수는 %d이며, 임계 레이블 순위는 %d인 상태로 진행합니다." % (BATCH_QUANTITY_FOR_EVAL, CNN.THRESHOLD_FOR_CDD))

    drift_count = 0  # 모든 시도에서 발생한 drift 횟수를 카운트하기 위한 변수

    # 최초 초기화
    batch_labels, batch_values = sess.run([Input.labels, Input.values])  # 입력을 가져옴

    vectorA = sess.run(CNN.y_conv, feed_dict={
        CNN.x: batch_values, CNN.y_: batch_labels, CNN.keep_prob: 1.0})  # 윈도우를 하나 가져옴

    last_label = vectorA

    for i in range(1, BATCH_QUANTITY_FOR_EVAL):  # 1개는 이미 받아왔으므로
        batch_labels, batch_values = sess.run([Input.labels, Input.values])  # 입력을 가져옴

        # correct_count_per_batch = sess.run(CNN.correct_answer, feed_dict={
        #    CNN.x: batch_values, CNN.y_:batch_labels, CNN.keep_prob: 1.0})
        concept_equal, vectorA = sess.run([CNN.correct_answer, CNN.y_conv], feed_dict={
            CNN.x: batch_values, CNN.y_: last_label, CNN.keep_prob: 1.0})

        '''        
        difference = np.abs(np.subtract(vectorA, vectorB)).sum()                # 두 벡터의 차이 벡터를 계산하고, 각 원소의 절대값을 모두 더함

        if(difference >= THRESHOLD):
            print("%d시점에서 CD 발생(difference = %g)"%(i, difference))
            drift_count += 1
            # print(vectorB)
        '''
        if (concept_equal == 0):
            print(i, "시점에서 concept drift 발생");
            drift_count += 1
        last_label = vectorA

    print("총 %d 개의 윈도우 중, %d회의 concept drift가 발생하였습니다." % (EVAL_DATA_QUANTITY, drift_count))

    #    print("정확도: %.2f%%" %(correct_count_total / (BATCH_QUANTITY_FOR_EVAL * Input.NUM_OF_BATCH) * 100))  # 최종 정확도 출력
    '''
    for i in range(100):
        examples = sess.run(nums)
        
        print(examples)
    '''
    coord.request_stop()
    coord.join(threads)
