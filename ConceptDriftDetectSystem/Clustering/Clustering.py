import csv
import numpy as np
from sklearn.cluster import KMeans

# import Input_Clustering as Input
'''
# with open('../options.ini', 'r') as csvfile:
csvfile = open("./Data/splitted_data.csv", 'r')     # 경로 바꿀 것
csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')

all_data = []

for row in csv_reader:
    row = list(map(int, row))
    all_data.append(row)

KMeans = KMeans(n_clusters=2, random_state=0)
KMeans.fit(all_data)
print(KMeans.labels_)
print(KMeans.cluster_centers_)
print(all_data)
'''

'''
def KMeans():
    X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    print(kmeans.labels_)
    print(kmeans.predict([[0, 0], [4, 4]]))
    print(kmeans.cluster_centers_)
'''

# 임의의 csv 파일을 받아 클러스터링을 수행하고, 클러스터러 객체를 반환한다
def KMeans_from_a_file(INPUT_FILE_PATH, CENTROIDS_SAVE_FILE_PATH, N_CLUSTERS):
    # with open('../options.ini', 'r') as csvfile:
    csvfile = open(INPUT_FILE_PATH, 'r')     # 경로 바꿀 것
    csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|', lineterminator='\n')

    all_data = []

    # csv 파일에서 데이터를 읽어옴
    for row in csv_reader:
        row = list(map(float, row))
        all_data.append(row)

    #‘k-means++’ : selects initial cluster centers 
    #              for k-mean clustering in a smart way to speed up convergence. 

    clustered = KMeans(n_clusters = N_CLUSTERS, init='k-means++', random_state = 0)
    #clustered = KMeans(n_clusters = N_CLUSTERS, init='k-means++', random_state = 1214, verbose = 1, n_init = 10, tol= 1e-18, n_jobs = -3)
    
    # Compute k-means clustering.
    clustered.fit(all_data)

    centroidfile = open(CENTROIDS_SAVE_FILE_PATH, 'w')
    csv_writer = csv.writer(centroidfile, delimiter=',', quotechar='|', lineterminator='\n')

    # write the coordition date of center of each clustering.
    for row in clustered.cluster_centers_:
        csv_writer.writerow(row)

    csvfile.close()
    centroidfile.close()

    return clustered

# 임의의 csv 파일과 레이블 리스트를 입력으로, CNN이 학습가능한 레이블이 포함된 csv 파일로 변환한다
# 차원 축소가 이루어지지 않은 윈도우 데이터에 레이블을 다는 것이 사용 예
def attatch_label_to_a_file(INPUT_FILE_PATH, OUTPUT_FILE_PATH, LABELS):
    in_file = open(INPUT_FILE_PATH, 'r')
    csv_reader = csv.reader(in_file, delimiter=',', lineterminator='\n')

    out_file = open(OUTPUT_FILE_PATH, 'w')
    csv_writer = csv.writer(out_file, delimiter=',', lineterminator='\n')

    i = 0
    for row in csv_reader:
        row = list(map(float, row))
        row.insert(0, LABELS[i])  # 윈도우 데이터의 가장 앞에 레이블 삽입
        csv_writer.writerow(row)
        i += 1

    in_file.close()
    out_file.close()
    pass