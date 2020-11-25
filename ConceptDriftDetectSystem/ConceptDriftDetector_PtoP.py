import configparser
import os
import numpy as np
import math
import timeit  # 실행시간 측정을 위한 모듈

import Experimenter  # precision, recall 계산을 위한 실험 모듈

# ini 설정을 불러오기 위한 변수 및 객체
CONF_FILE = "options.ini"
config = configparser.ConfigParser()
config.read(CONF_FILE)

# 전역 설정값 로드
section = "GENERAL"
DATA_FOLDER = "./" + config.get(section, 'DATA FOLDER NAME')  # 데이터 폴더는 공통으로 사용할 것

# 윈도우 정보
section = "STREAM DATA"  # ini 파일의 섹션
WINDOW_SIZE = int(config.get(section, 'WINDOW SIZE'))

# 윈도우 정보
section = "CLUSTERING"  # ini 파일의 섹션
CLUSTERS_FILE_PATH = DATA_FOLDER + "/" + config.get(section, 'CENTROIDS SAVE FILE NAME')

# 개념 변화 검출 관련 설정 값
section = "CONCEPT DRIFT DETECTION"
STREAM_DATA_PATH = DATA_FOLDER + "/" + config.get(section, 'STREAM DATA FILE NAME')
THRESHOLD_TO_TRIGGER = float(config.get(section, 'THRESHOLD TO TRIGGER'))
SLIDE_STEP_SIZE = int(config.get(section, 'SLIDE STEP SIZE'))
WINDOWS_AMOUNT_TO_VERIFY_PERSISTENT = int(config.get(section, 'WINDOWS AMOUNT TO VERIFY PERSISTENT'))


def overThresholdAmount(ARRAY, THRESHOLD):
    diffs = np.array(ARRAY)
    np.putmask(diffs, diffs >= THRESHOLD, 1)
    np.putmask(diffs, diffs < THRESHOLD, 0)
    return int(np.sum(diffs, 0))


experimenter = Experimenter.Experimenter(199, 1000)
start_time = timeit.default_timer()  # 시간 측정 시작

#################################################################################
# 1-1. 데이터 스트림 준비
import DataStream as DS

print("1.1 데이터 스트림을 준비합니다. 패러미터는", STREAM_DATA_PATH, WINDOW_SIZE, SLIDE_STEP_SIZE, "입니다.")
a_stream = DS.FileAsStream(STREAM_DATA_PATH, WINDOW_SIZE, SLIDE_STEP_SIZE)
print("데이터 스트림이 준비되었습니다.")

#################################################################################
# 1-2. Centroids 준비
from Clustering import ClusterCentroidsReader

print("1-2. Centroids 정보를 불러오는 중입니다...")
cluster_centroids, difference_matrix = ClusterCentroidsReader.readNCalc(CLUSTERS_FILE_PATH)
print("Centroids 정보를 모두 읽었습니다.")
print("클러스터 간 거리를 모두 사전 계산하였습니다.")

#################################################################################
# 1-3. CNN 준비

# CNN 관련 설정값
section = "CNN"  # ini 파일의 섹션
MODEL_FOLDER = "./" + config.get(section, 'MODEL SAVE FOLDER NAME') + "/"
# TRAIN_NUM = int(config.get(section, 'TRAIN NUM'))

print("1-3. 학습된 CNN을 로드합니다.")
import tensorflow as tf
from Autoencoder import Autoencoder_TF as AE  # 여기서 오토인코더를 사용하지는 않으나, 구조상의 문제로 일단 넣어야 동작함
from CNN import CNN_TF as CNN

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

#################################################################################
# 2. CNN을 통한 레이블 획득
import X as classX  # X 윈도우의 구성을 위한 모듈

X = classX.X(difference_matrix)  # X를 생성

with tf.Session(config=config) as sess:
    saver = tf.train.Saver()  # 학습이 끝난 모델 로드를 위한 saver 객체
    ckpt = tf.train.get_checkpoint_state(MODEL_FOLDER)
    if ckpt and ckpt.model_checkpoint_path:
        # tf.reset_default_graph()
        print("checkpoint 파일이 존재합니다. load 하겠습니다.")
        saver.restore(sess, MODEL_FOLDER)
    else:
        print("checkpoint 파일이 존재하지 않습니다. 프로그램을 종료합니다.")
        exit()
    print("checkpoint 파일의 load에 성공하였습니다.")

    # 윈도우들을 준비
    reference_window = a_stream.getWindow()
    dst_window = a_stream.getWindow()

    reference_window.label = sess.run(CNN.top_k_op, feed_dict={
        CNN.x: [reference_window.data], CNN.y_: [[0 for x in range(300)]], CNN.keep_prob: 1.0})[1][0][
        0]  # 최초 레퍼런스 윈도우에 대한 레이블을 설정

    concept_drift_counter = 0  # 개념 변화 카운터

    while (True):  # ::::::::::::::조건 나중에 변경해야 할 것임. 스트림의 끝을 어떻게 감지하지?::::::::::::::

        dst_window.label = sess.run(CNN.top_k_op, feed_dict={
            CNN.x: [dst_window.data], CNN.y_: [[0 for x in range(300)]], CNN.keep_prob: 1.0})[1][0][
            0]  # 비교 대상 윈도우에 대한 레이블을 설정

        # 아래 조건문에서 두 윈도우 사이에서 drift가 발생하는 지 우선 확인. 방법 1 수행과정 1에 해당
        if (difference_matrix[reference_window.label][dst_window.label] >= THRESHOLD_TO_TRIGGER):
            # 검증 과정을 위해 X 윈도우를 구성. X의 첫 번째 윈도우는 비교 대상 윈도우이며, 이후 X-1개의 윈도우가 들어차게 됨.

            X.clearExceptMatrix()  # 기존에 남아있는 요소들이 있을 수 있으므로, 비워버림
            X.append(dst_window.label, dst_window.time, dst_window.data)  # X의 첫 번째 윈도우는 비교 대상 윈도우

            for i in range(1, WINDOWS_AMOUNT_TO_VERIFY_PERSISTENT):
                temp = a_stream.getWindow()
                temp.label = sess.run(CNN.top_k_op, feed_dict={
                    CNN.x: [temp.data], CNN.y_: [[0 for x in range(300)]], CNN.keep_prob: 1.0})[1][0][
                    0]  # X에 들어갈 윈도우에 대한 레이블
                X.append(temp.label, temp.time, temp.data)

            # 아래에서 X내의 과반수가 레퍼런스 윈도우와 임계값 이상의 격차를 만들어내는 지 확인. 방법 1 수행과정 2에 해당
            diff_ref_X = X.differenceAgainstALabel(reference_window.label)
            # 레퍼런스 윈도우와 X 내의 윈도우들이 충분히 다른가?(과반수가 임계값을 넘겼는가? 윈도우가 8개였다면 5이상, 윈도우가 9개였다면 5이상)
            if (overThresholdAmount(diff_ref_X, THRESHOLD_TO_TRIGGER) > math.floor(X.length / 2)):

                # 아래에서 X내의 윈도우 간 서로 이웃하는 윈도우를 서로 비교. 방법 1 수행과정 3에 해당
                diff_neighbor = X.differenceFromNeighbors()
                # X 내의 윈도우들이 충분히 유사한가?(과반수가 임계값 이내인가? 윈도우가 8개였다면 5이하, 윈도우가 9개였다면 5이하)
                if (overThresholdAmount(diff_neighbor, THRESHOLD_TO_TRIGGER) <= math.ceil(X.length * 0.5)):
                    # 모든 검증이 완료된 것. drift가 실제로 발생했다고 판단하고 알림을 발생시키고 후속처리 수행

                    # 알림 발생
                    print(X.times[-1], "시점에서 Concept drift가 발생했습니다.")
                    experimenter.input(X.times[-1])

                    concept_drift_counter += 1

                    # X 내에서 가장 많이 발생한 레이블과, 그 레이블이 X에서 처음 등장한 인덱스를 찾아냄
                    most_label, most_label_index = X.mostNumorousLabelNFirstIndex()

                    # 다음 레퍼런스 윈도우는 X에서 가장 많이 발생한 레이블이 처음으로 등장한 윈도우
                    reference_window.label = X.labels[most_label_index]
                    reference_window.time = X.times[most_label_index]
                    reference_window.data = X.data[most_label_index]

        # print(dst_window.time)    # 끝부분 확인을 위한 디버깅 출력

        # 다음 비교 대상 윈도우는 이후 처음으로 등장하는 윈도우. 위 조건을 모두 만족했던, 일부라도 만족하지 않았던 수행한다
        try:
            dst_window = a_stream.getWindow()
        except:
            print("스트림 데이터를 읽어오는 중에 문제가 발생했습니다.")
            finish_time = timeit.default_timer()
            print("실행 시간:", finish_time - start_time)
            experimenter.result()
            exit()
        X.clearExceptMatrix()
