# -*- coding: utf-8 -*-

'''
Concept Drift 검출 모델을 다루기 위한 모듈이다.
이 모듈 내 함수들을 통해서만 검출 모델을 다루도록 한다.
'''
import shutil
import os  # 시스템 제어(폴더 복사 등)를 위해 사용
import pandas as pd
import configparser  # 모듈 복사 후 설정값 세팅에 사용
import sys
import urllib  # ftp를 통한 파일 다운로드에 사용

# ini 설정을 불러오기 위한 변수 및 객체
CONF_FILE = "server_options.ini"
config = configparser.ConfigParser()
config.read(CONF_FILE)

# 사용자가 제공하는 파일을 받기 위한 ftp 정보
section = "FTP SERVER CONNECT"  # ini 파일의 섹션
ftp_ip = config.get(section, 'IP')
ftp_port = config.get(section, 'PORT')
ftp_user = config.get(section, 'USER')
ftp_passwd = config.get(section, 'PASSWD')


def setWithKafka(id, topic, data_amount, dst_ip, dst_port):
    '''
    메시지를 처음 받았을 때 호출될 함수. id를 받아, 독립적으로 사용할 검출 모델을 하나 구성해준다.
    모델 구성시, 학습 데이터를 생성
    '''

    serial_num = id
    folder_name = str(serial_num)  # 폴더명은 호출한 pid를 따름

    if not os.path.exists(folder_name):
        print("데이터 폴더가 존재하지 않습니다. 새로 생성합니다.")  # 하위 모듈에서는 관심갖지 않으니, 미리 만들어둠
        os.makedirs(folder_name)

    shutil.copytree('ConceptDriftDetectSystem', folder_name)  # 폴더 복사

    ##########################
    # 모델의 복사는 끝냈으니 이하 config 수행. 혹시 몰라 다시 불러옴.
    ##########################
    # ini 설정을 불러오기 위한 변수 및 객체
    CONF_FILE = "./" + folder_name + "/options.ini"
    config = configparser.ConfigParser()
    config.read(CONF_FILE)

    # 스트림 데이터 관련 세팅
    section = "STREAM DATA"
    config.set(section, "KAFKA SERVER", str(kafkaserver))
    config.set(section, "KAFKA TOPIC", str(topic))
    config.set(section, "DATA AMOUNT", str(data_amount))

    conf_file = open(CONF_FILE, 'w')
    config.write(conf_file)
    conf_file.close()

    return 0


def setWithFileURL(id, src_name, topic, file_url):
    '''
    메시지를 처음 받았을 때 호출될 함수. 
    id를 받아, 독립적으로 사용할 검출 모델을 하나 구성해준다.
    모델 구성시, 학습 데이터를 생성
    '''

    folder_name = str(id + '.' + src_name)  # 폴더명은 호출한 유저명.소스이름

    if os.path.exists(folder_name):
        print("모델 폴더가 이미 존재합니다. 삭제하고 새로 생성합니다.")
        shutil.rmtree(folder_name)  # 폴더 삭제 함수

    shutil.copytree('ConceptDriftDetectSystem', folder_name)  # 폴더 복사

    ##########################
    # 모델의 복사는 끝냈으니 이하 config 수행
    ##########################
    # ini 설정을 불러오기 위한 변수 및 객체
    CONF_FILE = "./" + folder_name + "/options.ini"
    config = configparser.ConfigParser()
    config.read(CONF_FILE)

    # ftp를 통해 받을 파일 위치 지정을 위한 설정 값
    data_folder_name = config.get("GENERAL", 'DATA FOLDER NAME')
    train_data_file_name = config.get("STREAM DATA", 'STREAM DATA FILE NAME')  # 학습 데이터가 될 파일 이름

    # 학습 데이터를 FTP로부터 가져오기위하여 먼저 URL 정제
    splitted_url = file_url.split("/")
    file_name = splitted_url[-1]

    # 학습 데이터를 FTP로부터 가져옴
    # 받기 전에 먼저 data_folder_name을 생성 (디렉토리가 없으면 train_data_file이 생성되지 않음.)
    if not os.path.exists('{}/{}'.format(folder_name, data_folder_name)):
        os.makedirs('{}/{}'.format(folder_name, data_folder_name))

    ftp_url = 'ftp://' + ftp_user + ':' + ftp_passwd + '@' + ftp_ip + '/' + id + '/' + file_name
    urllib.request.urlretrieve(ftp_url, folder_name + '/' + data_folder_name + '/' + train_data_file_name)

    # 스트림 데이터 관련 세팅
    section = "GENERAL"
    config.set(section, "USER ID", str(id))
    config.set(section, "SOURCE NAME", str(src_name))

    # 스트림 데이터 관련 세팅
    section = "STREAM DATA"
    config.set(section, "KAFKA TOPIC", str(topic))

    conf_file = open(CONF_FILE, 'w')
    config.write(conf_file)

    conf_file.close()


def trainAModel(id, src_name):
    '''
    id 폴더 내의 데이터를 학습시킴
    '''
    folder_name = str(id + '.' + src_name)  # 폴더명은 호출한 유저명.소스이름

    # ini 설정을 불러오기 위한 변수 및 객체
    CONF_FILE = "./" + folder_name + "/options.ini"
    config = configparser.ConfigParser()
    config.read(CONF_FILE)

    # 전역 설정값 로드(이후에 사용할 것)
    section = "GENERAL"
    DATA_FOLDER = "./" + folder_name + "/" + config.get(section, 'DATA FOLDER NAME')  # 데이터 폴더는 공통으로 사용할 것

    if not os.path.exists(DATA_FOLDER):
        print("데이터 폴더가 존재하지 않습니다. 새로 생성합니다.")  # 하위 모듈에서는 관심갖지 않으니, 미리 만들어둠
        os.makedirs(DATA_FOLDER)

    # 데이터 학습을 시킴.
    os.chdir("./" + str(folder_name))
    os.system(sys.executable + " TrainSupervisor.py")
    os.chdir("..")


# ftp 에서 입력받은 훈련 파일을 일련의 정수형태로 정제한다.
def preprocessTrainfile(filename, real):
    # ini 설정을 불러오기 위한 변수 및 객체
    CONF_FILE = "./" + filename + "/options.ini"
    config = configparser.ConfigParser()
    config.read(CONF_FILE)

    # ftp를 통해 받을 파일 위치 지정을 위한 설정 값
    data_folder_name = config.get("GENERAL", 'DATA FOLDER NAME')
    train_data_file_name = config.get("STREAM DATA", 'STREAM DATA FILE NAME')  # 학습 데이터가 될 파일 이름
    os.chdir("./" + str(filename))

    df = pd.read_csv('./{}/{}'.format(data_folder_name, train_data_file_name), header=None)
    index = real[0]['COLUMN_INDEX']
    realValue = df.iloc[:, index]
    realValue.to_csv("./{}/temp.csv".format(data_folder_name), index=None, header=None)
    os.remove('./{}/{}'.format(data_folder_name, train_data_file_name))
    os.rename("./{}/temp.csv".format(data_folder_name), './{}/{}'.format(data_folder_name, train_data_file_name))
    os.chdir("..")


def setKafkaTopic(id, src_name, topic, real):
    folder_name = str(id + '.' + src_name)  # 폴더명은 호출한 유저명.소스이름
    index = real[0]['COLUMN_INDEX']  # comma 로 나눠진 value 중 어느 index 에 위치하는지

    if os.path.exists(folder_name):
        # ini 설정을 불러오기 위한 변수 및 객체
        CONF_FILE = "./" + folder_name + "/options.ini"
        config = configparser.ConfigParser()
        config.read(CONF_FILE)

        # 스트림 데이터 관련 세팅
        section = "GENERAL"
        config.set(section, "USER ID", str(id))
        config.set(section, "SOURCE NAME", str(src_name))

        # 스트림 데이터 관련 세팅
        section = "STREAM DATA"
        config.set(section, "KAFKA TOPIC", str(topic))
        config.set(section, "DATA INDEX", str(index))

        conf_file = open(CONF_FILE, 'w')
        config.write(conf_file)
        conf_file.close()
    else:
        print("Error : 모델 폴더({})가 존재하지 않습니다.".format(folder_name))
        return -1


def delete_model(model_name):
    '''
    모델 폴더들을 삭제하는 함수
    '''
    if os.path.exists(model_name):
        print("모델 폴더({})를 삭제합니다.".format(model_name))
        shutil.rmtree(model_name)  # 폴더 삭제 함수
    else:
        print("해당 모델 폴더({})가 없습니다.".format(model_name))


def detectStart(id):
    '''
    id 폴더의 모델을 이용하여 드리프트 검출을 수행
    '''
    if os.path.exists(id):
        os.chdir("./" + str(id))
        # 입력을 받아와야함. result = os.system(,,,
        os.system(sys.executable + " ConceptDriftDetector_PtoW.py")
        os.chdir("..")
    else:
        print("Error : 모델 폴더({})가 존재하지 않습니다.".format(id))
        return -1


def detectStop(id):
    '''
    id 폴더의 모델을 이용하여 드리프트 검출을 정지시킨다.
    '''

    query = "pkill -f {}".format("ConceptDriftDetector_PtoW.py")
    os.system(query)
    print("Shutdown {} complete".format("ConceptDriftDetector_PtoW"))


# 테스트 드라이버
if __name__ == '__main__':
    pass
