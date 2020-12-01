import tensorflow as tf

#############################################################################
# 이하 신경망 입력 제공부
#############################################################################
with tf.device('/cpu:0'):
    ROW_LENGTH = 121        # 레이블 포함
    NUM_OF_BATCH = 1

    # define filename queue
    filename_queue = tf.train.string_input_producer(["./data/concept_drift_detect.csv"])

    # define reader
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)

    #define decoder
    record_defaults = [[0.0]] * ROW_LENGTH
    a_row = tf.decode_csv(value, record_defaults=record_defaults)

    # define bulk decorder
    # labels, values = tf.train.batch([a_row[0], a_row[1:]], batch_size = NUM_OF_BATCH)   # 맨 앞의 값 하나가 레이블, 나머지는 데이터
    
    # values 처리
    temp_values = tf.to_float(a_row[1:])                                                    # 정규화 1단계, 실수변환

    # 아래는 정규화 과정인데, 하는 동작은 같음. 일단은 두 버전 모두 유지함

    # numpy 버전
    #temp_values = tf.div(temp_values, np.full(ROW_LENGTH - 1, 255))     # 정규화 2단계, 나눗셈 수행
    #temp_values = tf.add(temp_values, np.full(ROW_LENGTH - 1, 0.01))    # 정규화 2단계, 나눗셈 수행

    # tensorflow 버전
    #temp_values = tf.div(temp_values, tf.fill([ROW_LENGTH - 1], 255.0))   # 정규화 2단계, 나눗셈 수행
    #temp_values = tf.add(temp_values, tf.fill([ROW_LENGTH - 1], 0.01))    # 정규화 3단계, 0값을 막기위해 0.01을 더함

    # 최종 결과값
    scaled_values = temp_values
    label_vectors = tf.one_hot(tf.cast(a_row[0], tf.int32), depth = 101, on_value=0.99, off_value=0.01)

    labels, values = tf.train.batch([label_vectors, scaled_values], batch_size = NUM_OF_BATCH, num_threads=1)   # 맨 앞의 값 하나가 레이블, 나머지는 데이터. 순서를 유지한채로 입력데이터로 가져옴
    #labels, values = tf.train.shuffle_batch([label_vectors, scaled_values], batch_size = NUM_OF_BATCH, num_threads=4, capacity=10000, min_after_dequeue=5000)   # 맨 앞의 값 하나가 레이블, 나머지는 데이터. 스까
