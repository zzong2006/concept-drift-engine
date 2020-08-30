# -*- coding: utf-8 -*-

import configparser
import os

# ini 설정을 불러오기 위한 변수 및 객체
CONF_FILE = "options.ini"
config = configparser.ConfigParser()
config.read(CONF_FILE)

# 전역 설정값 로드
section = "GENERAL"
DATA_FOLDER = "./" + config.get(section, 'DATA FOLDER NAME')     # 데이터 폴더는 공통으로 사용할 것

#################################################################################
# 학습 단계
#################################################################################
# 1. 학습 데이터 준비
######################################
# 1-0. 먼저 관련 설정값을 불러온다
section = "STREAM DATA"  # ini 파일의 섹션

# 데이터 생성 관련 설정값
STREAM_DATA_PATH = DATA_FOLDER + "/" + config.get(section, 'STREAM DATA FILE NAME')
DATA_AMOUNT = int(config.get(section, 'DATA AMOUNT'))

# 데이터 분할 관련 설정값
WINDOW_DATA_PATH = DATA_FOLDER + "/" + config.get(section, 'WINDOW DATA FILE NAME')
WINDOW_SIZE = int(config.get(section, 'WINDOW SIZE'))
SLIDE_STEP_SIZE = int(config.get(section, 'SLIDE STEP SIZE'))



# # CNN 관련 설정값
# # 클러스터링 관련 설정값
# section = "CLUSTERING"  # ini 파일의 섹션
# CLUSTER_AMOUNT = int(config.get(section, 'CLUSTER AMOUNT'))     # 생성할 클러스터 수
# LABEL_ATTATCHED_WINDOW_DATA_PATH = DATA_FOLDER + "/" + config.get(section, 'LABEL ATTATCHED WINDOW DATA FILE NAME')
# CENTROIDS_SAVE_FILE_PATH = DATA_FOLDER + "/" + config.get(section, 'CENTROIDS SAVE FILE NAME')
#
# section = "CNN"  # ini 파일의 섹션
# MODEL_FOLDER = "./" + config.get(section, 'MODEL SAVE FOLDER NAME') + "/"
# TRAIN_NUM = int(config.get(section, 'TRAIN NUM'))
# KEEP_PROB = float(config.get(section, 'KEEP PROB'))
#
# from CNN import Train
#
# print("CNN 모델의 학습을 시작합니다. 패러미터는", MODEL_FOLDER, LABEL_ATTATCHED_WINDOW_DATA_PATH, int(config.get(section, 'BATCH SIZE')), TRAIN_NUM, int(config.get(section, 'NUM OF 1ST KERNEL')), int(config.get(section, 'NUM OF 2ND KERNEL')), int(config.get(section, 'L1 SIZE')),"입니다.\n")
# Train.train(MODEL_FOLDER, TRAIN_NUM, KEEP_PROB)
# pass
#

######################################
# 1-2. 스트림 데이터를 읽어 윈도우로 분할한다
print("1-2. 데이터를 윈도우로 분할합니다.")
from DataGenerator import StreamDataSplitter
num_of_window_data = 0  # 나뉜 윈도우의 갯수를 저장

print("윈도우 데이터를 구성합니다. 패러미터는", STREAM_DATA_PATH, WINDOW_DATA_PATH, WINDOW_SIZE, SLIDE_STEP_SIZE, "입니다.")
num_of_window_data = StreamDataSplitter.split_data_file_to_file(STREAM_DATA_PATH, WINDOW_DATA_PATH, WINDOW_SIZE, SLIDE_STEP_SIZE)
print("총 %d개의 윈도우 데이터를 구성하였습니다." % num_of_window_data)

print("")


#################################################################################
# 2. 오토인코더를 통한 데이터 축소
print("2. 오토인코더를 이용하여 데이터를 축소합니다.")

section = "DATA REDUCING"
MODEL_FOLDER = "./" + config.get(section, 'MODEL SAVE FOLDER NAME')
REDUCED_DATA_PATH = DATA_FOLDER + "/" + config.get(section, 'REDUCED DATA FILE NAME')

######################################
# 2-1. 오토인코더의 학습을 수행한다
print("2-1. 오토인코더를 학습시킵니다.")
from Autoencoder import Train

print("Autoencoder 모델의 학습을 시작합니다. 패러미터는", MODEL_FOLDER, int(config.get(section, 'TRAIN NUM')),
      int(config.get(section, 'BATCH SIZE')), int(config.get(section, 'HIDDEN LAYER #1')),
      int(config.get(section, 'REDUCED DIMENSION')),"입니다.\n")
Train.train()
pass

print("")

######################################
# 2-2. 학습된 오토인코더를 이용하여 모든 윈도우 데이터를 축소한다
print("2-2. 학습된 오토인코더를 이용하여 모든 윈도우 데이터를 축소합니다.")

from Autoencoder import Datareduce

print("Autoencoder를 통한 데이터 축소를 진행합니다. 패러미터는", REDUCED_DATA_PATH, int(config.get(section, 'BATCH SIZE')),"입니다.")
Datareduce.datareduce(num_of_window_data, REDUCED_DATA_PATH)    # 이를 통해 REDUCED_DATA가 생성됨
pass

print("")


#################################################################################
# 3. k-means를 이용한 클러스터링
print("3. 클러스터링(K-Means)을 이용하여 모든 윈도우 데이터에 레이블을 부여합니다.")

# 클러스터링 관련 설정값
section = "CLUSTERING"  # ini 파일의 섹션
CLUSTER_AMOUNT = int(config.get(section, 'CLUSTER AMOUNT'))     # 생성할 클러스터 수
LABEL_ATTATCHED_WINDOW_DATA_PATH = DATA_FOLDER + "/" + config.get(section, 'LABEL ATTATCHED WINDOW DATA FILE NAME')
CENTROIDS_SAVE_FILE_PATH = DATA_FOLDER + "/" + config.get(section, 'CENTROIDS SAVE FILE NAME')

from Clustering import Clustering

# 3-1. 파일을 읽어 클러스터링 수행
print("3-1. 클러스터링을 수행합니다. 패러미터는", REDUCED_DATA_PATH, CENTROIDS_SAVE_FILE_PATH, CLUSTER_AMOUNT, "입니다.")
KMeans = Clustering.KMeans_from_a_file(REDUCED_DATA_PATH, CENTROIDS_SAVE_FILE_PATH, CLUSTER_AMOUNT)   # 임의 속성 개수를 갖는 csv 데이터를 읽어 클러스터링을 수행하는 모듈. 생성할 클러스터 수는 여기서 지정한다.

# 3-2. 클러스터링 결과를 윈도우 데이터에 붙임
print("3-2. 클러스터링을 통해 얻은 레이블을 윈도우 데이터와 합칩니다. 패러미터는", WINDOW_DATA_PATH, LABEL_ATTATCHED_WINDOW_DATA_PATH, "입니다.")
Clustering.attatch_label_to_a_file(WINDOW_DATA_PATH, LABEL_ATTATCHED_WINDOW_DATA_PATH, KMeans.labels_)
pass

print("")

#################################################################################
# 4. CNN 학습을 수행한다

# CNN 관련 설정값
section = "CNN"  # ini 파일의 섹션
MODEL_FOLDER = "./" + config.get(section, 'MODEL SAVE FOLDER NAME') + "/"
TRAIN_NUM = int(config.get(section, 'TRAIN NUM'))
KEEP_PROB = float(config.get(section, 'KEEP PROB'))

print("4-1. CNN의 학습을 수행합니다.")

from CNN import Train

print("CNN 모델의 학습을 시작합니다. 패러미터는", MODEL_FOLDER, LABEL_ATTATCHED_WINDOW_DATA_PATH, int(config.get(section, 'BATCH SIZE')), TRAIN_NUM, int(config.get(section, 'NUM OF 1ST KERNEL')), int(config.get(section, 'NUM OF 2ND KERNEL')), int(config.get(section, 'L1 SIZE')),"입니다.\n")
Train.train(MODEL_FOLDER, TRAIN_NUM, KEEP_PROB)
pass

print("")

'''
print("4-2. 학습된 CNN을 평가합니다.")

# 모델 폴더가 존재하는 지 확인
if os.path.exists(MODEL_FOLDER):
    YorN = input(">>CNN 모델의 평가는 선택사항입니다. 학습데이터에 대하여 모델을 평가하시겠습니까?(y/N): ")
    if(YorN == 'Y' or YorN == 'y'):
        from CNN import Eval
        Eval.eval(MODEL_FOLDER, num_of_window_data)        
    else:
        print("모델의 평가를 생략하고 진행합니다.")
        pass
else:   # 폴더가 없는 경우
    input("학습된 CNN을 찾는데 문제가 발생했습니다. 프로그램을 종료합니다. ")
    pass
'''

#from CNN import Eval
#Eval.eval(MODEL_FOLDER, num_of_window_data)


'''
검출 단계(모듈 분리됨)
'''

# 1. 데이터 스트림 준비

# 2. CNN을 통한 레이블 획득

# 3. Concept drift 검출(Label adjust)