import random
import numpy as np

'''
TrainDataGenerator(StreamDataGeneratorV4)
대규모, 일부값을 조정 가능한 실험을 위하여 v2 버전과 다르게 데이터의 생성 및 분할 모듈을 분리하여 구현한 버전
이 모듈은 긴 데이터를 생성하는 역할을 한다. 윈도우의 개념은 여기서 나타나지 않는다.
V4버전은 출력 형태가 바뀌었으며 호출가능한 모듈형태로 변경하였다. 리스트를 출력한다.
'''

# 파일 및 데이터 규모 관련 상수 선언부
FILE_NAME_PREFIX = ''  # 't10k' or 'train'
FILE_NAME = "raw_data.raw"
GEN_FILE_QUANTITY = 1  # 생성할 파일 개수
GEN_DATA_QUANTITY = 5000  # 생성할 데이터의 개수

# 단일 데이터 관련 상수 선언부
START_DATA = 122  # 시작 데이터
VAR_MEAN = 0.0  # 변위로 사용할 정규분포의 평균
VAR_SIGMA = 4.0  # 변위로 사용할 정규분포의 표준편차
MAX_NUM = 255  # 단일 데이터가 가질 수 있는 최대값
MIN_NUM = 0  # 단일 데이터가 가질 수 있는 최소값


# 1개의 데이터를 생성하여 리턴하는 함수
# 데이터의 생성방식 및 변위가 어떻게 될 지 고정되어있지 않기 때문에 함수로 따로 빼두어 수정이 용이하게 한다.
def make_a_data(last_data):
    varience = int(random.gauss(VAR_MEAN, VAR_SIGMA))  # 단일 데이터에 가할 변위
    next_number = last_data + varience  # 데이터에 변위를 가함

    if next_number > MAX_NUM or next_number < MIN_NUM:  # 단일 데이터의 도메인 범위를 벗어난 경우, 오버플로우를 막기위해 범위 내 값으로 조정한다. 변위의 부호만 반전시키면 된다.
        next_number = last_data - varience
    return next_number


# 전체 데이터를 생성하는 함수
# 데이터는 하나의 리스트로 구성되며, 함수는 이를 반환한다
def generate_data(GEN_DATA_QUANTITY, START_DATA, VAR_MEAN, VAR_SIGMA):
    #######################################################################################################
    ''' MAIN '''
    stream_data = []  # 전체 스트림 데이터가 저장될 배열

    gen_number = START_DATA
    output = str(gen_number)

    stream_data.append(output)

    # 데이터를 1개씩 추가로 생성해가며(변위를 주어가며) 파일에 출력
    for x in range(GEN_DATA_QUANTITY - 1):
        gen_number = make_a_data(gen_number)
        output = str(gen_number)
        stream_data.append(output)
    return stream_data


def generate_data_to_file(FILE_NAME, GEN_DATA_QUANTITY, START_DATA, VAR_MEAN, VAR_SIGMA):
    ''' MAIN '''
    # 스트림 데이터 생성 시작
    for file_no in range(1, GEN_FILE_QUANTITY + 1):
        # file_name = 'data_batch_%d.bin' % file_no
        out_file = open(FILE_NAME_PREFIX + FILE_NAME, 'w')

        gen_number = START_DATA
        output = str(gen_number)
        out_file.write(output + '\n')  # 변위가 없는 데이터를 일단 하나 파일에 기록

        # 데이터를 1개씩 추가로 생성해가며(변위를 주어가며) 파일에 출력
        for x in range(GEN_DATA_QUANTITY - 1):
            gen_number = make_a_data(gen_number)
            output = str(gen_number)
            out_file.write(output + '\n')
            if (x % (GEN_DATA_QUANTITY / 10) == 0 and x != 0):
                print(GEN_DATA_QUANTITY, "중 ", x, "개의 데이터 생성 완료")  # 식이 좀 이상한데, 어짜피 진행율을 보이는 것이니 그다지 상관없음
        print("총", x + 2, "개의 데이터 생성 완료")  # 2 = 앞에 미리 만든 1개 + 인덱스가 0부터 시작
        out_file.close()
        return x
    pass


#######################################################################################################
if __name__ == "__main__":
    generate_data_to_file(FILE_NAME, GEN_DATA_QUANTITY, START_DATA, VAR_MEAN, VAR_SIGMA)
    pass
