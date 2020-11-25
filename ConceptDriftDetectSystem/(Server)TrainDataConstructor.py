'''
임의의 kafka의 임의 토픽에서 데이터를 지정된 수만큼 받아, 학습 데이터를 생성하는 모듈
'''
from kafka import KafkaConsumer
import configparser
import timeit  # 실행시간 측정을 위한 모듈


def Train_data_construct(server, topic, file_path, data_amount):
    consumer = KafkaConsumer(topic, bootstrap_servers=[server],
                             auto_offset_reset='earliest',
                             max_poll_records=1000,
                             consumer_timeout_ms=3000
                             )

    received_data = []
    out_file = open(file_path, "w")

    start_time = timeit.default_timer()  # 시간 측정 시작
    print("학습을 위한 데이터를 kafka(" + server + ", " + topic + ")로부터" + str(data_amount) + " 만큼 전달받습니다.")
    counter = 0
    # consumer.subscribe([topic])

    while (counter < data_amount):
        temp = consumer.poll()
        if temp != {}:
            records = list(temp.values())
            values = [int(i.value) for i in records[0]]
            received_data.extend(values)  # The method extend() appends the contents of seq to list.
            counter = counter + len(values)

    finish_time = timeit.default_timer()
    print("kafka 전송 소요 시간:", finish_time - start_time)

    print("학습 데이터가 모두 모였습니다. 학습 파일의 생성을 수행합니다.")
    for num in received_data:
        out_file.write(str(num) + '\n')
    exit()


if __name__ == "__main__":
    # ini 설정을 불러오기 위한 변수 및 객체
    CONF_FILE = "options.ini"
    config = configparser.ConfigParser()
    config.read(CONF_FILE)

    # 전역 설정값 로드
    section = "GENERAL"
    DATA_FOLDER = "./" + config.get(section, 'DATA FOLDER NAME')  # 데이터 폴더는 공통으로 사용할 것

    # 스트림 정보
    section = "STREAM DATA"  # ini 파일의 섹션
    server = config.get(section, 'KAFKA SERVER')
    # topic = config.get(section, 'KAFKA TOPIC')
    topic = 'test_stream'
    file_path = DATA_FOLDER + "/" + config.get(section, 'STREAM DATA FILE NAME')
    data_amount = int(config.get(section, 'DATA AMOUNT'))

    Train_data_construct(server, topic, file_path, data_amount)
