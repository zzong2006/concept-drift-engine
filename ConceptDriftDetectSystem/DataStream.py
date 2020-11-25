'''
DataStreamV2
스트림 파일을 하나 지정하여 이를 윈도우 단위로 접근할 수 있도록 만든 모듈이다.
이후 파일이 아닌 네트워크에서 입력되는 데이터에도 대응 가능하도록 변경이 용이하게 만들었다.
'''
import Window
import time
import numpy as np

class Unit:
    '''
    스트림에서 하나의 데이터를 저장하는 단위인 Unit 클래스
    '''
    def __init__(self, value, offset= -1, partition=-1):
        self.value = value
        self.offset = offset
        self.partition = partition
        pass

class Stream:
    '''
    각 하위 클래스들의 중복된 코드들을 담는 부모 클래스.
    이 클래스만으로는 데이터를 가져오지 못하기 때문에 실제 스트림의 역할을 하지 못한다.
    '''
    def __init__(self, window_size, step_size):
        '''
        윈도우 관련 변수 초기화
        '''
        self.time = 0
        self.count = 0
        self.window_size = window_size
        self.step_size = step_size
        pass

    def getWindow(self):
        '''
        현재 차있는 윈도우를 반환하는 함수. 반환 후 slideWindow 함수를 호출한다.
        '''
        out_window = Window.Window(self.window, self.time)
        out_window.data = (out_window.data - np.mean(out_window.data))/np.std(out_window.data)
        self.slideWindow()
        return out_window
        pass

    def slideWindow(self):
        '''
        윈도우를 이동하는 함수. init 당시의 step_size 만큼 이동한다
        '''
        self.window = self.window[self.step_size:]
        self.window.extend([self.getData() for i in range(self.step_size)])
        self.time += 1
        pass

    def getData(self):
        '''
        하나의 데이터를 스트림으로부터 가져오는 함수.
        자식 클래스에서 오버라이딩 되어야 함.
        '''
        a_data = 0
        return a_data
        pass


class FileAsStream(Stream):
    '''
    파일을 읽어 이를 스트림으로 사용하는 클래스.
    과거의 클래스로, 현재의 클래스와 호환되지 않을 것(offset 관련).
    '''
    def __init__(self, file_name, window_size, step_size):
        '''
        윈도우를 초기화하는 함수
        '''
        Stream.__init__(self, window_size, step_size)

        self.data_file = open(file_name, 'r')
        self.window = [self.getData() for i in range(window_size)]
        pass

    def getData(self):
        '''
         1개의 데이터를 받아오는 함수. 
        '''
        '''
        try:
            a_value = int(self.data_file.readline())
            a_value = self.normalize(a_value)
            return a_value
        except:     # 예외 처리는 우선 뭉뚱그려서 처리했음
            print("스트림 데이터를 읽어오는 중에 문제가 발생했습니다.")
            exit()
        '''
        a_data = Unit(float(self.data_file.readline().rstrip('\n')))
        val += 1
        # a_data.value = self.normalize(a_data.value)
        return a_data
        pass

class KafkaAsStream(Stream):
    def __init__(self, stream_addr, topic, window_size, step_size, index):
        '''
        윈도우를 초기화하는 함수.
        '''
        from kafka import KafkaConsumer

        Stream.__init__(self, window_size, step_size)
        self.timeout = 10
        self.index = index
        self.consumer = KafkaConsumer(topic, bootstrap_servers=stream_addr,
                                auto_offset_reset='earliest',
                                max_poll_records = 1,
                                consumer_timeout_ms=5000
                                )
        # Unit class의 Data list를 window 변수에 담는다.
        self.window = [self.getData() for i in range(window_size)]
    def getData(self):
        count = 0
        while(count <= 10):
            temp = self.consumer.poll()
            if temp != {}:
                records = list(temp.values())
                tempData = (records[0][0].value).decode('utf-8')
                refinedData = [x.strip() for x in tempData.split(',')]
                a_data = Unit(float(refinedData[self.index]), int(records[0][0].offset), int(records[0][0].partition))
                return a_data
            else:
                print('Kafka Server 에서 data 를 기다리는 중.. ({}/{} sec)'.format(count, self.timeout))
                count += 2
                time.sleep(2)       # 2초후에 다시 시도
        raise ValueError("Kafka Server 에서 10초 동안 data 입력이 없습니다.")

if(__name__ == '__main__'):
    a_stream = FileAsStream("./Data/stream1.raw", 1000, 40)
    data = a_stream.getWindow()
    print(data.data)
    '''
    

    data = a_stream.getWindow()
    #print(data.data)
    ''' 

    # b_stream = KafkaAsStream("165.132.214.219:9093", "test_stream", 1000, 40)
    # data2 = b_stream.getWindow()
    # print(data2.data)