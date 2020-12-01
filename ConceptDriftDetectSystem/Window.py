'''
윈도우 객체 클래스를 정의하는 모듈이다. DataStream에서 사용하는 모듈이며, 직접 실행할 일은 없을 것.
'''

import numpy as np


class Window:
    # 윈도우를 초기화하는 함수
    def __init__(self, data, time):
        self.time = time  # 윈도우가 생성된 시점을 의미 0부터 카운트
        self.data = [i.value for i in data]
        self.offset = [i.offset for i in data]
        self.partition = [i.partition for i in data]
        # self.offset = offsets # kafka에서 각 데이터에 대해 받았을 오프셋
        self.label = -1  # 신경망에서 정보를 얻기 전에는 모름


    def setLabel(self, label):
        self.label = label


    # 직접 접근해도 되나, 필요하다면 쓸 수 있는 함수
    def getData(self):
        return self.data


    def getTime(self):
        return self.time


    def getLabel(self):
        return self.label

