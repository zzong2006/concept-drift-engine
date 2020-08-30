import tensorflow as tf
import configparser
import sys

CONF_FILE = "options.ini"
config = configparser.ConfigParser()
config.read(CONF_FILE)

# 전역 설정값 로드
section = "GENERAL"
DATA_FOLDER = "./" + config.get(section, 'DATA FOLDER NAME')     # 데이터 폴더는 공통으로 사용할 것

section = "STREAM DATA"
ROW_LENGTH = int(config.get(section, 'WINDOW SIZE')) + 1

section = "CLUSTERING"
LABEL_ATTATCHED_WINDOW_DATA_FILE_NAME = config.get(section, 'LABEL ATTATCHED WINDOW DATA FILE NAME')
CLUSTER_AMOUNT = int(config.get(section, 'CLUSTER AMOUNT'))
DATA_PATH = DATA_FOLDER + "/" + LABEL_ATTATCHED_WINDOW_DATA_FILE_NAME
# DATA_PATH = os.path.abspath(DATA_PATH)

section = "CNN"
BATCH_SIZE = int(config.get(section, 'BATCH SIZE'))

# ROW_LENGTH = 121        # 레이블 포함
# NUM_OF_BATCH = 200;

#############################################################################
# 이하 신경망 입력 제공부
#############################################################################
with tf.device('/cpu:0'):
    # define filename queue
    filename_queue = tf.train.string_input_producer([DATA_PATH], shuffle=False, name='filename_queue')

    # define reader
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    # value = tf.Print(value, [value], "AI Adventures Says:")

    #define decoder
    record_defaults = [[0.0]] * ROW_LENGTH
    a_row = tf.decode_csv(value, record_defaults=record_defaults)

    # define bulk decorder
    # labels, values = tf.train.batch([a_row[0], a_row[1:]], batch_size = BATCH_SIZE)   # 맨 앞의 값 하나가 레이블, 나머지는 데이터
    
    # values 처리
    temp_values = tf.to_float(a_row[1:])

    temp_values_s0 = tf.to_float(temp_values)  # 정규화 1단계, 실수변환

    mean, var = tf.nn.moments(temp_values_s0, axes=[0])
    scaled_values = (temp_values_s0 - mean) / tf.sqrt(var)

    label_vectors = tf.one_hot(tf.cast(a_row[0], tf.int32), depth = CLUSTER_AMOUNT, on_value=1, off_value=0)
    labels, values = tf.train.batch([label_vectors, scaled_values], batch_size = BATCH_SIZE, num_threads=1)

    #  맨 앞의 값 하나가 레이블, 나머지는 데이터. 순서를 유지한채로 입력데이터로 가져옴
    # labels, values = tf.train.shuffle_batch([label_vectors, scaled_values],
    #                                         batch_size = BATCH_SIZE,
    #                                         num_threads=4,
    #                                         capacity=10000,
    #                                         min_after_dequeue=5000)
    # 맨 앞의 값 하나가 레이블, 나머지는 데이터. 스까
    # labels = tf.Print(labels, [labels], "AI Adventures Says:")
    # values= tf.Print(values, [values], "AI Adventures Says:")
