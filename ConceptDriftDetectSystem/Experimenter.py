import numpy as np


class Experimenter:
    def __init__(self, drift_amount, drift_interval):
        self.detect_counter = [0 for x in range(drift_amount + 1)]
        self.detect_delay = [0 for x in range(drift_amount + 1)]
        self.drift_interval = drift_interval
        self.drift_amount = drift_amount
        pass

    def input(self, detect_time):
        target_slot = int(detect_time / self.drift_interval)
        self.detect_counter[target_slot] += 1
        if self.detect_delay[target_slot] == 0:
            self.detect_delay[target_slot] = (detect_time % self.drift_interval)
        pass

    def result(self):
        # 모델 검출 결과 분석
        # 맨 첫 구역은 1개라도 검출되면 false alarm인 것에 주의
        correct_alarm = np.array(self.detect_counter[1:])
        np.putmask(correct_alarm, correct_alarm > 0, 1)
        correct_alarm = sum(correct_alarm)
        print("Currect Alarm:", correct_alarm)

        false_alarm = sum(self.detect_counter) - correct_alarm
        print("False Alarm:", false_alarm)

        false_dismissal = self.detect_counter[1:].count(0)
        print("False Dismissal:", false_dismissal)

        precision = correct_alarm / (correct_alarm + false_alarm)
        print("Precision:", precision)

        recall = correct_alarm / (correct_alarm + false_dismissal)
        print("Recall:", recall)

        # 검출 딜레이 계산
        temp_arr = [x for x in self.detect_delay[1:] if x != 0]
        # temp_arr.remove(0)
        average = sum(temp_arr) / len(temp_arr)
        print("Avg Delay to Detection:", average)


if __name__ == '__main__':
    experimenter = Experimenter(199, 1000)
    experimenter.input(10)
    experimenter.input(11)
    experimenter.input(12)
    experimenter.input(1001)
    experimenter.input(1002)
    experimenter.input(1003)
    experimenter.input(1004)

    experimenter.result()
