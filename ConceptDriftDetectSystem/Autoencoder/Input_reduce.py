import tensorflow as tf
import configparser
import os

CONF_FILE = "options.ini"
config = configparser.ConfigParser()
config.read(CONF_FILE)

# 전역 설정값 로드
section = "GENERAL"
DATA_FOLDER = "./" + config.get(section, 'DATA FOLDER NAME')     # 데이터 폴더는 공통으로 사용할 것

section = "STREAM DATA"
ROW_LENGTH = int(config.get(section, 'WINDOW SIZE'))     # 데이터 폴더는 공통으로 사용할 것
WINDOW_DATA_FILE_NAME = config.get(section, 'WINDOW DATA FILE NAME')
DATA_PATH = DATA_FOLDER + "/" + WINDOW_DATA_FILE_NAME
# DATA_PATH = os.path.abspath(DATA_PATH)

section = "DATA REDUCING"
BATCH_SIZE = int(config.get(section, 'BATCH SIZE'))

#############################################################################
# 이하 신경망 입력 제공부
#############################################################################
with tf.device('/cpu:0'):

    # define filename queue
    filename_queue = tf.train.string_input_producer([DATA_PATH])

    # define reader
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)

    #define decoder
    record_defaults = [[0.0]] * ROW_LENGTH
    a_row = tf.decode_csv(value, record_defaults=record_defaults)

    # define bulk decorder
    # labels, values = tf.train.batch([a_row[0], a_row[1:]], batch_size = NUM_OF_BATCH)   # 맨 앞의 값 하나가 레이블, 나머지는 데이터

    # values 처리
    temp_values_s0 = tf.to_float(a_row)  # 정규화 1단계, 실수변환

    mean, var = tf.nn.moments(temp_values_s0, axes=[0])
    scaled_values = (temp_values_s0 - mean) / tf.sqrt(var)

    # 최종 결과값
    #label_vectors = tf.one_hot(tf.cast(a_row[0], tf.int32), depth = 101, on_value=0.99, off_value=0.01) # 어짜피 쓰이지 않음

    # with tf.device('/gpu:0'):
    # labels, values = tf.train.batch([label_vectors, scaled_values], batch_size = BATCH_SIZE, num_threads=1)   # 맨 앞의 값 하나가 레이블, 나머지는 데이터. 순서를 유지한채로 입력데이터로 가져옴
    # labels, values = tf.train.shuffle_batch([label_vectors, scaled_values], batch_size = NUM_OF_BATCH, num_threads=4, capacity=10000, min_after_dequeue=5000)   # 맨 앞의 값 하나가 레이블, 나머지는 데이터. 스까

    values = tf.train.batch([scaled_values], batch_size = BATCH_SIZE, num_threads=1)   # 맨 앞의 값 하나가 레이블, 나머지는 데이터. 순서를 유지한채로 입력데이터로 가져옴. 스레드 수 늘리지 말 것. 순서가 바뀜
    #values = tf.train.shuffle_batch([scaled_values], batch_size = BATCH_SIZE, num_threads=4, capacity=3000, min_after_dequeue=1000)   # 맨 앞의 값 하나가 레이블, 나머지는 데이터. 스까