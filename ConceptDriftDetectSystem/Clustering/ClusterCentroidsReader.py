import numpy as np
import csv
import os


# 클러스터 중심점 정규화 함수. 현재는 사용하지 않는 함수
def normalizeaRow(array):
    out_array = np.array(array) / 255.0
    return out_array


def read(in_file_path):
    in_file = open(in_file_path, 'r')
    in_file_csv = csv.reader(in_file, delimiter=',', quotechar='|', lineterminator='\n')

    cluster_centroids = []

    '''
    for line in in_file_csv:
        for i in range(len(line)):
            line[i] = int(line[i])
            pass
        centroids.append(line)
        pass
    '''

    # csv 파일에서 데이터를 읽어옴
    for row in in_file_csv:
        row = list(map(float, row))
        row = normalizeaRow(row)  # 정규화
        cluster_centroids.append(row)

    return cluster_centroids


def readNCalc(in_file_path):
    in_file = open(in_file_path, 'r')
    in_file_csv = csv.reader(in_file, delimiter=',', quotechar='|', lineterminator='\n')

    cluster_centroids = []

    '''
    for line in in_file_csv:
        for i in range(len(line)):
            line[i] = int(line[i])
            pass
        centroids.append(line)
        pass
    '''

    # csv 파일에서 데이터를 읽어옴
    for row in in_file_csv:
        row = list(map(float, row))
        cluster_centroids.append(row)

    matrix = []

    # 클러스터의 거리를 사전 계산하여 매트리스를 얻음
    for i in range(len(cluster_centroids)):
        temp_matrix = np.array([cluster_centroids[i] for a in range(len(cluster_centroids))])
        # temp_matrix = np.abs(np.subtract(temp_matrix, cluster_centroids))
        temp_matrix = np.power(np.subtract(temp_matrix, cluster_centroids), 2)  # 각 원소의 차이를 제곱
        matrix.append(temp_matrix)
    difference_matrix = np.array(matrix).sum(axis=2)  # 모든 클러스터 센트로이드 조합의 경우의 수에 대한 거리를 담을 매트릭스
    difference_matrix = np.sqrt(difference_matrix)  # 각 차이를 합산한 결과에 루트
    # matrix = matrix.sum(axis=1)

    # 이하 정규화
    # difference_matrix = difference_matrix + abs(np.amin(difference_matrix))    # 사용하지 않는게 나을 것 같아 제거. 값들을 너무 편향되게 만듬
    difference_matrix = difference_matrix / np.amax(
        difference_matrix)  # 클러스터 거리 매트릭스 모든 값들을 [0, 1] 사이의 수로 매핑. 당연히 전체 매트릭스 내에서 크기의 대소관계는 유지된다.

    return cluster_centroids, difference_matrix  # 솔직히 클러스터 센트로이드를 검출모델에서 직접 쓸 일이 있을지는 의문이나, 일단 기능은 남겨둠


#######################################################################################################
if __name__ == "__main__":
    in_file_path = open('./cluster_centroids.csv', 'r')
    cluster_centroids, difference_matrix = readNCalc("./cluster_centroids.csv")
    pass
