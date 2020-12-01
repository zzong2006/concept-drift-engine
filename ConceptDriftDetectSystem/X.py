'''
PtoW를 구현하기 위해 윈도우 X 를 구현한 클래스이다.
persistent 속성을 구분하기 위하여 사용하는 리스트 세트. 조금 특이한 형태의 queue 자료형이라 생각하면 된다.
Window 객체를 보관하게 할까 싶기도 했으나, 참조가 번거로워지고 중복 import를 야기하므로 객체의 속성만을 리스트들로 관리하기로 했다
X는 윈도우 데이터 자체는 보관하지 않는다
'''
import numpy as np


def over_threshold_amount(ARRAY, THRESHOLD):
    diffs = np.array(ARRAY)
    np.putmask(diffs, diffs >= THRESHOLD, 1)
    np.putmask(diffs, diffs < THRESHOLD, 0)
    return int(np.sum(diffs, 0))


class X:
    # 윈도우를 초기화하는 함수
    def __init__(self, difference_matrix):
        self.labels = []
        self.times = []
        self.data = []
        self.offsets = []
        self.partitions = []
        self.difference_between_index0_with_all = []
        self.difference_matrix = difference_matrix
        self.length = 0
        pass

    def append(self, label, time, data, offset, partition):
        self.labels.append(label)
        self.times.append(time)
        self.data.append(data)
        self.offsets.append(offset)
        self.partitions.append(partition)
        self.length += 1
        self.difference_between_index0_with_all.append(self.difference_matrix[self.labels[0]][label])
        pass

    def delete(self):
        if (self.length > 0):
            self.labels.remove(self.labels[0])
            self.times.remove(self.times[0])
            self.data.remove(self.data[0])
            self.offsets.remove(self.offsets[0])
            self.partitions.remove(self.partitions[0])
            self.length -= 1
            if (self.length > 0):  # 제거 후에 원소가 없을 수도 있기 때문에 아래와같이 추가 분기 수행
                self.difference_between_index0_with_all = self.differenceAgainstALabel(self.labels[0])
            else:
                self.difference_between_index0_with_all = []
        else:
            pass
        pass

    def clearExceptMatrix(self):
        self.labels.clear()
        self.times.clear()
        self.data.clear()
        self.offsets.clear()
        self.partitions.clear()
        self.length = 0
        self.difference_between_index0_with_all = []
        pass

    # 1개 객체분량의 데이터를 받아서 한 칸씩 밀어내는 함수
    def slide(self, label, time, data, offset, partition):
        self.delete()
        self.append(label, time, data, offset, partition)
        pass

    ### concept drift 검출을 위한 함수들. 인스턴스의 값을 변경하는 일은 없으며, 없어야 함
    # X 내의 특정 윈도우와 전체 원소의 비교값을 리스트로 만들어 반환하는 함수. 인덱스로 특정 윈도우를 지정가능하다.
    def differenceAgainstALabel(self, ref_label):
        difference_list = []
        for l in self.labels:
            difference_list.append(self.difference_matrix[ref_label][l])
        return difference_list

    # X 내에서 서로 이웃하는 윈도우 간의 비교값을 리스트로 만들어 반환하는 함수. 
    # index 0은 맨 첫번째와 맨 마지막 레이블을 비교한 결과이며, 나머지 위치는 직전 윈도우와 비교. 사실 순서는 중요하지 않음
    def differenceFromNeighbors(self):
        difference_list = []
        for l in range(len(self.labels)):
            difference_list.append(self.difference_matrix[self.labels[l - 1]][self.labels[l]])
        return difference_list

    # X 내에서 가장 많은 레이블이 무엇인지 찾아, 그 레이블과 그 레이블이 나타나는 첫 번째 인덱스를 반환하는 함수
    # 레이블이 총 몇 종류인지는 difference_matrix를 통해 알 수 있다
    def mostNumorousLabelNFirstIndex(self):
        count_array = [0 for i in range(len(self.difference_matrix[0]))]  # 카운트를 위한 리스트 선언
        # 카운트
        for label in self.labels:
            count_array[label] += 1
        most_label = np.argmax(count_array)
        return most_label, self.labels.index(most_label)


# 테스트 드라이버
if (__name__ == '__main__'):
    import configparser

    # ini 설정을 불러오기 위한 변수 및 객체
    CONF_FILE = "options.ini"
    config = configparser.ConfigParser()
    config.read(CONF_FILE)

    # 전역 설정값 로드
    section = "GENERAL"
    DATA_FOLDER = "./" + config.get(section, 'DATA FOLDER NAME')  # 데이터 폴더는 공통으로 사용할 것

    # 윈도우 정보
    section = "CLUSTERING"  # ini 파일의 섹션
    CLUSTERS_FILE_PATH = DATA_FOLDER + "/" + config.get(section, 'CENTROIDS SAVE FILE NAME')

    # Centroids 준비
    from Clustering import ClusterCentroidsReader

    cluster_centroids, difference_matrix = ClusterCentroidsReader.readNCalc(CLUSTERS_FILE_PATH)

    X = X(difference_matrix)
    # X.append(1, 0, [1])
    # X.append(2, 1, [1])
    # X.append(3, 2, [1])
    # X.append(5, 3, [1])
    # X.append(1, 3, [1])
    # X.append(1, 3, [1])
    # X.append(5, 3, [1])
    # X.append(5, 4, [1])
    # X.append(5, 4, [1])
    most_label, most_label_index = X.mostNumorousLabelNFirstIndex()
    # X.slide(6, 7, [1,2])
    X.delete()
    X.delete()
    X.delete()
    X.delete()
    X.delete()
    X.clearExceptMatrix()
