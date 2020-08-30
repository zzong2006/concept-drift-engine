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
START_DATA = int(config.get(section, 'START DATA'))
MEAN = float(config.get(section, 'VAR MEAN'))
SIGMA = float(config.get(section, 'VAR SIGMA'))


# 데이터 분할 관련 설정값
WINDOW_DATA_PATH = DATA_FOLDER + "/" + config.get(section, 'WINDOW DATA FILE NAME')
WINDOW_SIZE = int(config.get(section, 'WINDOW SIZE'))
SLIDE_STEP_SIZE = int(config.get(section, 'SLIDE STEP SIZE'))

#################################################################################
# 2. 오토인코더를 통한 데이터 축소
print("2. 오토인코더를 이용하여 데이터를 축소합니다.")

section = "DATA REDUCING"
MODEL_FOLDER = "./" + config.get(section, 'MODEL SAVE FOLDER NAME')
REDUCED_DATA_PATH = DATA_FOLDER + "/" + config.get(section, 'REDUCED DATA FILE NAME')


from Autoencoder import test

test.train()
