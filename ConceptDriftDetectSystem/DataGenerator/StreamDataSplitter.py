'''
    데이터를 window 단위로 나눠주는 splitter.
'''

import csv

def split_data_file_to_file(INPUT_FILE_NAME='data.csv', OUTPUT_FILE_NAME='splitted_data.csv', WINDOW_SIZE=1000, SLIDE_STEP_SIZE=200):
    '''MAIN'''
    with open(INPUT_FILE_NAME, 'r') as in_file, open(OUTPUT_FILE_NAME, 'w') as out_file:
        out_file_csv = csv.writer(out_file, delimiter=',', quotechar='|', lineterminator='\n')

        # while True:
        #     print('slide step 단위로 윈도우 생성? (1) 또는 window size 단위로 윈도우 생성? (2)')
        #     y = int(input())
        #     if y == 1 or y == 2:
        #         break
        y = 1
        temp = in_file.read().splitlines()
        print('The whole number of data is ' + str(len(temp)))
        if len(temp) < WINDOW_SIZE:
            print('current window size is' + str(WINDOW_SIZE) + '. But your file has ' + len(temp) + ' data.')
            return -1
        num = 1

        if y == 1:
            num += int((len(temp) - WINDOW_SIZE) / SLIDE_STEP_SIZE)  # 총 생성할 window 개수

            if (len(temp) - WINDOW_SIZE) % SLIDE_STEP_SIZE > 0:
                print(
                    'There will be a window having ' + str(int((len(temp) - WINDOW_SIZE) % SLIDE_STEP_SIZE)) + ' data')
                print('Current window size is ' + str(WINDOW_SIZE))
            curr = 0

            for i in range(num):
                out_file_csv.writerow(temp[curr:(curr + WINDOW_SIZE)])
                curr += SLIDE_STEP_SIZE
        if y == 2:
            num += int((len(temp) - WINDOW_SIZE) / WINDOW_SIZE)  # 총 생성할 window 개수

            if (len(temp) % WINDOW_SIZE) > 0:
                print('There will be a window having ' + str(len(temp) % WINDOW_SIZE) + ' data')
                print('Current window size is ' + str(WINDOW_SIZE))
            curr = 0

            for i in range(num):
                out_file_csv.writerow(temp[curr:(curr + WINDOW_SIZE)])
                curr += WINDOW_SIZE
        return num

# csv 파일 내에 행이 몇 개 있는지 계산하여 반환하는 함수. 기존에 생성된 윈도우 데이터를 사용할 때 필요하다.
def how_many_windows_in_file(INPUT_FILE_NAME):
    '''MAIN'''
    in_file = open(INPUT_FILE_NAME, 'r')
    in_file_csv = csv.reader(in_file, delimiter=',', quotechar='|', lineterminator='\n')

    row_count = 0

    for data in in_file_csv:
        row_count += 1
    in_file.close()
    return row_count
