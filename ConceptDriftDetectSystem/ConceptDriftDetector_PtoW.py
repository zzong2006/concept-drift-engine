# 외부 모듈
import configparser
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import timeit  # 실행시간 측정을 위한 모듈

# 통신을 위한 모듈
import socket
import json

# import Experimenter # precision, recall 계산을 위한 실험 모듈

# ini 설정을 불러오기 위한 변수 및 객체
CONF_FILE = "options.ini"
config = configparser.ConfigParser()
config.read(CONF_FILE)

# 전역 설정값 로드
section = "GENERAL"
DATA_FOLDER = "./" + config.get(section, 'DATA FOLDER NAME')  # 데이터 폴더는 공통으로 사용할 것
USER_ID = config.get(section, 'USER ID')
SOURCE_NAME = config.get(section, 'SOURCE NAME')

# 스트림 정보
section = "STREAM DATA"  # ini 파일의 섹션
WINDOW_SIZE = int(config.get(section, 'WINDOW SIZE'))
SERVER = config.get(section, 'KAFKA SERVER')
INDEX = int(config.get(section, 'DATA INDEX'))
# 서버가 여러개일 경우 각각의 서버 ip:port 조합을 String list로 담는것이 중요하다.
SERVER = SERVER.split(',')
TOPIC = config.get(section, 'KAFKA TOPIC')

# 윈도우 정보
section = "CLUSTERING"  # ini 파일의 섹션
CLUSTERS_FILE_PATH = DATA_FOLDER + "/" + config.get(section, 'CENTROIDS SAVE FILE NAME')

# 개념 변화 검출 관련 설정 값
section = "CONCEPT DRIFT DETECTION"
STREAM_DATA_PATH = DATA_FOLDER + "/" + config.get(section, 'STREAM DATA FILE NAME')
THRESHOLD_TO_TRIGGER = float(config.get(section, 'THRESHOLD TO TRIGGER'))
SLIDE_STEP_SIZE = int(config.get(section, 'SLIDE STEP SIZE'))
WINDOWS_AMOUNT_TO_VERIFY_PERSISTENT = int(config.get(section, 'WINDOWS AMOUNT TO VERIFY PERSISTENT'))

# 개념 변화 검출 결과 수신 서버 설정 값
section = "DETECT RESULT RECV CLIENT"
DST_IP = config.get(section, 'IP')
DST_PORT = config.get(section, 'PORT')


def overThresholdAmount(ARRAY, THRESHOLD):
    diffs = np.array(ARRAY)
    np.putmask(diffs, diffs >= THRESHOLD, 1)
    np.putmask(diffs, diffs < THRESHOLD, 0)
    return int(np.sum(diffs, 0))


def transmit_result(offset, partition):
    try:
        s = socket.socket()
        s.connect((DST_IP, int(DST_PORT)))

        request = {"message": "concept-drift", "user-id": USER_ID, "src-name": SOURCE_NAME, "offset": offset,
                   "partition": partition}
        d_request = json.dumps(request)

        s.send(d_request.encode('utf-8'))
        s.close()
    except:
        s.close()
        pass
    return 0


def draw_serial_data(strData, currData, colorRange, current):
    fig, ax = plt.subplots()
    idx = np.arange(len(strData))
    color = np.array(['black' for _ in range(len(strData))], dtype=object)
    for al in colorRange:
        color[(idx >= al[0]) & (idx < al[1])] = "red"

    points = np.column_stack((idx, strData)).reshape(-1, 1, 2)
    segments = np.hstack([points[:-1], points[1:]])
    coll = LineCollection(segments, colors=color)
    ax.add_collection(coll)
    ax.autoscale_view()
    plt.show()


# experimenter = Experimenter.Experimenter(199, 1000)
# start_time = timeit.default_timer()     # 시간 측정 시작

#################################################################################
# 1-1. 데이터 스트림 준비
import DataStream as DS

# stream_addr, topic, window_size, step_size
print("1.1 데이터 스트림을 준비합니다. 패러미터는", SERVER, TOPIC, WINDOW_SIZE, SLIDE_STEP_SIZE, "입니다.")

# a_stream = DS.FileAsStream(STREAM_DATA_PATH, WINDOW_SIZE, SLIDE_STEP_SIZE, INDEX) # 로컬 테스트용
a_stream = DS.KafkaAsStream(SERVER, TOPIC, WINDOW_SIZE, SLIDE_STEP_SIZE, INDEX)  # 서버용
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
plottingColor = np.array([])

with tf.Session(config=config) as sess:
    saver = tf.train.Saver()  # 학습이 끝난 모델 로드를 위한 saver 객체
    ckpt = tf.train.get_checkpoint_state(MODEL_FOLDER)
    if ckpt and ckpt.model_checkpoint_path:
        print("checkpoint 파일이 존재합니다. load 하겠습니다.")
        saver.restore(sess, MODEL_FOLDER)
    else:
        print("checkpoint 파일이 존재하지 않습니다. 프로그램을 종료합니다.")
        exit()
    print("checkpoint 파일의 load에 성공하였습니다.")

    # 레퍼런스 윈도우와 X 윈도우를 구성
    reference_window = a_stream.getWindow()
    # for serial data plot
    global_serialData = np.array(reference_window.data)
    curr = 0

    reference_window.label = sess.run(CNN.top_k_op, feed_dict={
        CNN.x: [reference_window.data], CNN.keep_prob: 1.0})[1][0][0]  # 최초 레퍼런스 윈도우에 대한 레이블을 설정

    X.clearExceptMatrix()  # 기존에 남아있는 요소들이 있을 수 있으므로, 비워버림

    # 최초의 X 윈도우를 구성
    for i in range(WINDOWS_AMOUNT_TO_VERIFY_PERSISTENT):
        temp = a_stream.getWindow()
        temp.label = sess.run(CNN.top_k_op, feed_dict={
            CNN.x: [temp.data], CNN.keep_prob: 1.0})[1][0][0]  # X에 들어갈 윈도우에 대한 레이
        X.append(temp.label, temp.time, temp.data, temp.offset, temp.partition)
        global_serialData = np.append(global_serialData[:curr], X.data[-1])
        curr += SLIDE_STEP_SIZE
        # print("{} : {} ".format(i, temp.data))
    # print(len(X.data))

    # 이 시점에서 레퍼런스 윈도우 및 X 준비 완료
    while True:  # ::::::::::::::조건 나중에 변경해야 할 것임. 스트림의 끝을 어떻게 감지하지?::::::::::::::
        # 페이즈 1
        # 아래 과정에서 레퍼런스 윈도우와 X 내 과반수의 차이가 임계값을 넘겼는지 확인. 방법 2 수행과정 1에 해당
        diff_ref_X = X.differenceAgainstALabel(reference_window.label)
        # 레퍼런스 윈도우와 X 내의 윈도우들이 충분히 다른가?(과반수가 임계값을 넘겼는가? 윈도우가 8개였다면 5이상, 윈도우가 9개였다면 5이상) 달랐다면 페이즈2로 넘어감
        # print(overThresholdAmount(diff_ref_X, THRESHOLD_TO_TRIGGER))
        if overThresholdAmount(diff_ref_X, THRESHOLD_TO_TRIGGER) > math.floor(X.length / 2):
            # print(X.times[0], "시점에서 현재 concept과 크게 다른 concept 구간이 등장했습니다. drift가 시작되었으며, 지금부터 concept이 안정화되는 구간을 찾습니다.")
            # 페이즈 2 시작
            while True:
                # 아래 과정에서 X의 선두 윈도우와 X 내 과반수의 차이가 임계값 이내인지 확인. 방법 2 수행과정 2에 해당
                diff_X1_X = X.differenceAgainstALabel(X.labels[-1])
                # 레퍼런스 윈도우와 X 내의 윈도우들이 충분히 다른가?(과반수가 임계값을 넘겼는가? 윈도우가 8개였다면 5이상, 윈도우가 9개였다면 5이상) 달랐다면 페이즈2로 넘어감
                if overThresholdAmount(diff_X1_X, THRESHOLD_TO_TRIGGER) <= math.ceil(X.length / 2):
                    # print(X.times[0], "시점에서 안정화 된 concept 구간이 등장했습니다. drift가 종료되었으며, 지금부터 drift가 발생하는 구간을 찾습니다.")
                    print(X.times[-1], "시점에서 개념 변화가 검출되었습니다.. offset은", X.offsets[-1][-1], "이며, partition은",
                          X.partitions[-1][-1], "입니다.")
                    if plottingColor.size == 0:
                        plottingColor = np.array([[curr - SLIDE_STEP_SIZE, curr + WINDOW_SIZE - SLIDE_STEP_SIZE]])
                    else:
                        plottingColor = np.vstack((plottingColor,
                                                   np.array(
                                                       [curr - SLIDE_STEP_SIZE, curr + WINDOW_SIZE - SLIDE_STEP_SIZE])))
                    transmit_result(X.offsets[-1][-1], X.partitions[-1][-1])
                    draw_serial_data(global_serialData, X.data[-1], plottingColor, curr)

                    # 변화한 concept을 기준으로 다시 drift 검출을 하기 위하여 레퍼런스 윈도우를 X의 최후 윈도우로 지정
                    reference_window.label = X.labels[-1]
                    reference_window.time = X.times[-1]
                    reference_window.data = X.data[-1]
                    # X 윈도우를 비우기 위해 새로운 윈도우를 채워나감. 마지막 하나는 break문 이후에 1회 수행될 것이다.
                    for i in range(WINDOWS_AMOUNT_TO_VERIFY_PERSISTENT - 1):
                        try:
                            next_window = a_stream.getWindow()
                        except:
                            print("검출을 종료합니다.")
                            # finish_time = timeit.default_timer()
                            # print("실행 시간:", finish_time - start_time)
                            # experimenter.result()
                            exit()
                        next_window.label = sess.run(CNN.top_k_op, feed_dict={
                            CNN.x: [next_window.data], CNN.keep_prob: 1.0})[1][0][0]  # X에 들어갈 윈도우에 대한 레이블
                        X.slide(next_window.label, next_window.time, next_window.data, next_window.offset,
                                next_window.partition)
                        global_serialData = np.append(global_serialData[:curr], X.data[-1])
                        curr += SLIDE_STEP_SIZE
                    break  # 페이즈 2 루프를 끊어 페이즈 1으로 돌아감

                # X의 슬라이드를 위해 다음 윈도우를 가져옴
                try:
                    next_window = a_stream.getWindow()
                except:
                    print("검출을 종료합니다.")
                    # finish_time = timeit.default_timer()
                    # print("실행 시간:", finish_time - start_time)
                    exit()
                next_window.label = sess.run(CNN.top_k_op, feed_dict={
                    CNN.x: [next_window.data], CNN.keep_prob: 1.0})[1][0][0]  # X에 들어갈 다음 윈도우에 대한 레이블
                # X를 슬라이드 후 루프
                X.slide(next_window.label, next_window.time, next_window.data, next_window.offset,
                        next_window.partition)
                global_serialData = np.append(global_serialData[:curr], X.data[-1])
                curr += SLIDE_STEP_SIZE
            pass

        # X의 슬라이드를 위해 다음 윈도우를 가져옴
        try:
            next_window = a_stream.getWindow()
        except:
            print("검출을 종료합니다.")
            # finish_time = timeit.default_timer()
            # print("실행 시간:", finish_time - start_time)
            # experimenter.result()
            exit()
        next_window.label = sess.run(CNN.top_k_op, feed_dict={
            CNN.x: [next_window.data], CNN.keep_prob: 1.0})[1][0][0]  # X에 들어갈 다음 윈도우에 대한 레이블

        # X를 슬라이드 후 루프
        X.slide(next_window.label, next_window.time, next_window.data, next_window.offset, next_window.partition)
        global_serialData = np.append(global_serialData[:curr], X.data[-1])
        curr += SLIDE_STEP_SIZE
