# -*- coding: utf-8 -*-

'''
외부 모듈
'''
import configparser
import socket
import json  # JSON 인코딩 및 디코딩을 위한 모듈
import pymysql
from multiprocessing import Process  # 멀티 프로세싱 처리를 위함. os 상관없이 동작
import urllib.request

'''
내부 모듈
'''
import ManageDriver as MD

# ini 설정을 불러오기 위한 변수 및 객체
CONF_FILE = "server_options.ini"
config = configparser.ConfigParser()
config.read(CONF_FILE)

# 전역 설정값 로드
section = "MESSAGE RECEIVE"
msg_port = int(config.get(section, 'PORT'))

# 공용 저장소 DB 설정값 로드
section = "SHARED STORAGE SQL DB"  # ini 파일의 섹션
db_ip = config.get(section, 'IP')
db_port = int(config.get(section, 'PORT'))
db_user = config.get(section, 'USER')
db_passwd = config.get(section, 'PASSWD')
db_db = config.get(section, 'DB')
db_charset = config.get(section, 'CHARSET')


def getFromDB(a_json):
    '''
    원격지 DB에 접속하여, 입력된 메시지 내 사용자 및 소스에 해당하는 열 정보을 반환하는 함수
    '''
    ###################################
    # DB 질의 준비. 커넥션엔 타임아웃이 있으므로, 여기서는 매번 새롭게 연결한다.
    conn = pymysql.connect(host=db_ip, port=db_port, user=db_user, password=db_passwd, db=db_db, charset=db_charset)
    curs = conn.cursor(pymysql.cursors.DictCursor)

    ###############################################
    # 소스 정보 읽기: user-id와 src-name을 사용하여 TBL_SRC$IDX를 읽고 (현재는 user-id만 읽게 설정해놓음.),
    # 이를 사용하여 TBL_INTELLIGENT_ENGINE$F_TARGET 값을 읽는다.
    # F_TARGET으로 TBL_SRC_CSV_SCHEMA$IDX에 접근하여 타겟 컬럼 정보를 읽는다.
    #########
    preSql = 'select * from tbl_user where ID="' + a_json['user-id'] + '"'
    curs.execute(preSql)
    result00 = curs.fetchall()
    user_idx = result00[0]['IDX']

    sql = 'select * from tbl_src where F_OWNER="' + str(user_idx) + '" and NAME="' + str(a_json['src-name']) + '"'
    curs.execute(sql)
    result = curs.fetchall()
    user_src_idx = result[0]['IDX']

    # tbl_intelligent_engine은 지능형 알고리즘 추천 엔진 정보 테이블
    sql2 = 'select * from tbl_intelligent_engine where F_SRC="' + str(user_src_idx) + '"'
    curs.execute(sql2)
    result2 = curs.fetchall()
    if result2 == ():
        print('tbl_intelligent_engine 테이블에 USER: {}의 정보가 존재하지 않습니다.'.format(a_json['user-id']))
        result3 = None
    else:
        f_target = result2[0]['F_TARGET']
        sql3 = 'select * from tbl_src_csv_schema where IDX="' + str(f_target) + '"'
        curs.execute(sql3)
        result3 = curs.fetchall()

    '''
    example of 'result3' : target column information
    {'IDX': 43,
      'COLUMN_INDEX': 1,
      'COLUMN_NAME': 'data',
      'COLUMN_TYPE': 'NUMERIC',
      'F_SRC': 26}
    '''

    if (a_json['message'] == 'new-src'):
        # 소스 예제 파일 읽기(FTP): TBL_SRC$F_TEST_DATA로 TBL_SRC_TEST_DATA에 접근하여 FILE_PATH를 읽고 파일을 가져온다.
        sql4 = 'select * from tbl_src where NAME="' + str(a_json['src-name']) + '" and ' + 'F_OWNER="' + str(
            user_idx) + '"'
        curs.execute(sql4)
        result4 = curs.fetchall()
        f_test_data = result4[0]['F_TEST_DATA']

        # 이 소스가 학습데이터로 사용하고자 하는 파일의 경로를 알고자 함
        sql5 = 'select * from tbl_src_test_data where IDX="' + str(f_test_data) + '" and ' + 'F_OWNER="' + str(
            user_idx) + '"'
        curs.execute(sql5)
        result5 = curs.fetchall()
        file_path = result5[0]['FILE_PATH']

        conn.close()
        return result3, file_path

    if (a_json['message'] == 'activate-src'):
        # 이 소스의 kafka topic 이름을 알고자 함.
        sql6 = 'select * from tbl_src where NAME="' + str(a_json['src-name']) + '" and ' + 'F_OWNER="' + str(
            user_idx) + '"'
        curs.execute(sql6)
        result6 = curs.fetchall()
        topic_name = result6[0]['TRANS_TOPIC']

        conn.close()
        return result3, topic_name

    conn.close()
    return result3


# 모델 생성 유무 저장: TBL_SRC$CONCEPT_DRIFT_STATUS 값을 모델링 생성 유무에 따라 ‘PREPARED’ 또는 ‘NOT_USED’로 수정한다.
def change_status(a_json, result):
    conn = pymysql.connect(host=db_ip, port=db_port, user=db_user, password=db_passwd, db=db_db, charset=db_charset)
    curs = conn.cursor(pymysql.cursors.DictCursor)

    try:
        sql = 'select * from tbl_user where ID="' + a_json['user-id'] + '"'
        curs.execute(sql)
        out = curs.fetchall()
        user_idx = out[0]['IDX']

        if (result != -1):
            query = 'PREPARED'
        else:
            query = 'NOT_USED'

        sql = "UPDATE tbl_src SET CONCEPT_DRIFT_STATUS='%s' WHERE NAME='%s' and F_OWNER='%s'" % (
            query, str(a_json['src-name']), str(user_idx))
        logSql = "INSERT INTO tbl_log(F_USER,LOGGING_TYPE,LOGGING_MESSAGE) VALUES({},{},{})".format(user_idx, '"INFO"',
                                                                                                    '"[CONCEPT DRIFT]Model of source({}) status is changed to {}"'.format(
                                                                                                        a_json[
                                                                                                            'src-name'],
                                                                                                        query))

        curs.execute(logSql)
        curs.execute(sql)
        conn.commit()
    except Exception as ex:
        print('Error: change_status', ex)
    finally:
        conn.close()


# log type : 1; make model, 2; concept activate, 3; concept stop, 4; model delete
def sendLog(a_json, logtype):
    conn = pymysql.connect(host=db_ip, port=db_port, user=db_user, password=db_passwd, db=db_db, charset=db_charset)
    curs = conn.cursor(pymysql.cursors.DictCursor)

    try:
        sql = 'select * from tbl_user where ID="' + a_json['user-id'] + '"'
        curs.execute(sql)
        out = curs.fetchall()
        user_idx = out[0]['IDX']
        conn.commit()

        if logtype == 1:
            log_sql = "INSERT INTO tbl_log(F_USER,LOGGING_TYPE,LOGGING_MESSAGE) VALUES({},{},{})".format(user_idx,
                                                                                                         '"INFO"',
                                                                                                         '"[CONCEPT DRIFT]Model of source({}) is generated."'.format(
                                                                                                             a_json[
                                                                                                                 'src-name']))
        elif logtype == 2:
            log_sql = "INSERT INTO tbl_log(F_USER,LOGGING_TYPE,LOGGING_MESSAGE) VALUES({},{},{})".format(user_idx,
                                                                                                         '"INFO"',
                                                                                                         '"[CONCEPT DRIFT]Concept Drift Detection (Source: {}) is activated."'.format(
                                                                                                             a_json[
                                                                                                                 'src-name']))
        elif logtype == 3:
            log_sql = "INSERT INTO tbl_log(F_USER,LOGGING_TYPE,LOGGING_MESSAGE) VALUES({},{},{})".format(user_idx,
                                                                                                         '"INFO"',
                                                                                                         '"[CONCEPT DRIFT]Concept Drift Detection (Source: {}) is stopped."'.format(
                                                                                                             a_json[
                                                                                                                 'src-name']))
        elif logtype == 4:
            log_sql = "INSERT INTO tbl_log(F_USER,LOGGING_TYPE,LOGGING_MESSAGE) VALUES({},{},{})"
            log_sql = log_sql.format(user_idx,
                                     '"INFO"',
                                     '"[CONCEPT DRIFT]Model of source({}) is deleted."'.format(
                                         a_json[
                                             'src-name']))

        curs.execute(log_sql)
        conn.commit()
    except Exception as ex:
        print('Error: sendLog', ex)
    finally:
        conn.close()


def determine_n_proc(a_json):
    '''
    임의의 json 형식 메시지를 받아 메시지에 해당하는 동작을 수행하는 모듈
    '''
    # 컨셉드리프트 분석 모델 생성: 소스 예제 파일에서 타겟 컬럼을 기반으로 컨셉 드리프트 모델을 생성한다.
    if a_json['message'] == "new-src":  # SET(kafkaservers, topic, dst_ip_port)
        # print("새로운 모듈을 세트하기 위하여 DB에서 사용자 정보를 가져옵니다.")
        real_value, test_data_path = getFromDB(a_json)
        topic_name = 'NULL'
        result = MD.setWithFileURL(a_json['user-id'], a_json['src-name'], topic_name, test_data_path)
        change_status(a_json, result)
        if result != -1:
            MD.preprocessTrainfile(a_json['user-id'] + "." + a_json['src-name'], real_value)
            MD.trainAModel(a_json['user-id'], a_json['src-name'])
        sendLog(a_json, 1)
        # 소스 정보가 없기 때문에, 일단 하드 코딩
    # 데이터 스트림 읽기: 데이터 스트림에서 타겟 컬럼을 추출하여 컨셉 드리프트를 분석한다.
    elif a_json['message'] == "activate-src":  # SET(kafkaservers, topic, dst_ip_port)
        sendLog(a_json, 2)
        real_value, topic_name = getFromDB(a_json)
        result = MD.setKafkaTopic(a_json['user-id'], a_json['src-name'], topic_name, real_value)
        if result != -1:
            MD.detectStart(a_json['user-id'] + "." + a_json['src-name'])  # 모델을 통하여 검출 시작
    # 데이터 스트림 읽기 중단: 데이터 스트림을 읽지 않고 컨셉 드리프트 분석을 중단한다.
    elif a_json['message'] == "deactivate-src":
        _ = getFromDB(a_json)
        MD.detectStop(a_json['user-id'] + "." + a_json['src-name'])
        sendLog(a_json, 3)
    elif a_json['message'] == "destroy-src":
        MD.delete_model(a_json['user-id'] + "." + a_json['src-name'])
        sendLog(a_json, 4)
        # change_status(a_json, -1) # change to "NOT USED", Plan-Manager 에서 DB를 지워주기 때문에 상태를 바꾸지 않아도 될듯.
    else:
        print('wrong message ! : {}'.format(a_json))


# 메인 루프
if __name__ == '__main__':
    ###################################
    # 메인 루프 전처리 작업
    ###################################
    # 메시지 소켓 통신 준비

    ipmessage = urllib.request.urlopen("http://checkip.dyndns.org").read().decode("utf-8")
    external_ip = ''.join(c for c in ipmessage if c.isdigit() or c == '.')

    # inner ip address = "127.0.0.1"
    # external ip address = '165.132.214.219'
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 소켓 최적화 함수
    s.bind((external_ip, msg_port))  # 소켓 바인딩
    s.listen(5)

    connection_list = []

    # 메인 루프 시작
    while True:
        if len(connection_list) == 0:
            # Halts
            print('[Waiting for connection...]')
            c, addr = s.accept()  # (socket object, address info) return
            print('Got connection from', addr)
            connection_list.append(c)
        else:
            # Halts
            print('[Waiting for response...]')
            received = json.loads(c.recv(1024).decode('utf-8'))
            print(received)
            p = Process(target=determine_n_proc, args=(received,))  # 프로세스 하나를 생성
            p.start()  # 시작. 실행은 시키되, 프로세스의 실행 이후는 관여하지 않음. 이후 종료시키거나 관리하려면 따로 저장해야할 것.
            # determine_n_proc(received) # 디버깅용. 위 두줄을 비활성화하고 할 것
            c.close()
            connection_list.remove(c)
