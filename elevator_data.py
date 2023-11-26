import torch
import numpy as np
import urllib.request
import cv2
import datetime
import csv
import os

# 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)


#이미지 추출 당시 시간
timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S, %f")[:-3]  # 밀리세컨드까지


csv_path = ""
# 로그 파일 생성
def make_csv():
  global csv_path
  csv_path = f"data_csv/{timestamp}.csv"
  header = ['date', 'floor', 'arrow']
  try:
      with open(csv_path, 'x', newline='') as file:
          writer = csv.writer(file)
          writer.writerow(header)
  except FileExistsError:
      pass

# 저장 폴더 설정
save_folder = 'unidentified_images'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# 현재 층수, 화살표 저장을 위한 변수 (여기에 안둬도 되나....? 안둬도 될듯)
# floor = -1
# arrow = -1
# 이전 층수, 화살표 저장을 위한 변수
last_floor = -1
last_arrow = -1

#서버 주소
url = "http://172.16.102.134:8080/shot.jpg"

#클래스 name
class_name = ['B2', 'B1', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'DOWN', 'UP']

#층수가 입력되지 않았을 경우 이미지를 저장하기 위한 변수
unidentified_image_cnt = 15

#서버 문제 발생 여부
error = 3000

#이미지 파일 이름
filename = ''


# 이미지를 url주소로부터 추출하는 함수
def capture_image():
    global timestamp
    try:    
        with urllib.request.urlopen(url) as resp:
            image_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        timestamp = datetime.datetime.now().strftime("%m%d, %H%M%S, %f")[:-3]
        infer_and_find(image)

    except urllib.request.URLError as e:
        error -= 1
        if error > 2998:
            with open(csv_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, "URLError"])

    except urllib.request.HTTPError as e:
        error -= 1
        if error > 2998:
            with open(csv_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, "HTTPError"])


# 모델을 통해 추론 & floor, arrow 파악해서 처리하는 함수
def infer_and_find(img):
  results =model(img)

  #이미지 속 모든 클래스와 confidence 저장
  #result_class = [cls for t in results.xyxy for cls in t[:, 5].tolist()]
  #result_conf = [conf for t in results.xyxy for conf in t[:, 4].tolist()]
  result_class = []
  result_conf = []
  for t in results.xyxy:
    result_class.extend(t[:, 5].tolist())
    result_conf.extend(t[:, 4].tolist())

  #카테고리별 클래스 확인 (여러 클래스가 있다면 confidence가 가장 높은 값으로)
  floor = -1
  arrow = -1
  highest_floor_conf = -1
  highest_arrow_conf = -1

  for cls, conf in zip(result_class, result_conf):
      if cls < 11:
          # floor 값 중 가장 높은 confidence 값을 저장
          if conf > highest_floor_conf:
              highest_floor_conf = conf
              floor = cls
      else:
          # arrow 값 중 가장 높은 confidence 값을 저장
          if conf > highest_arrow_conf:
              highest_arrow_conf = conf
              arrow = cls

  if highest_floor_conf < 0.75:
     return
  #print('----------',floor, highest_floor_conf, '/', arrow, highest_arrow_conf)

  global unidentified_image_cnt
  global last_floor
  global last_arrow

  if floor < 0:  # 층수가 인식되지 않았을 때 -> 현재 이미지 저장
    if unidentified_image_cnt > 0:  # 이미지 저장 최대 갯수를 초과하기 전
      # 이미지 파일로 저장
      filename = f"{save_folder}/{timestamp}.jpg"
      cv2.imwrite(filename, img)
      unidentified_image_cnt -= 1
      last_floor = floor
      last_arrow = arrow
      return
    else:  # 이미지 저장 최대 갯수를 초과한 후 (더이상 이미지를 저장하지 X)
      return

  else:  # 층수가 인식되었을 때 -> 이미지 저장하지 않고 로그 기록 여부 체크
    if unidentified_image_cnt < 15:
      unidentified_image_cnt = 15 # 다시 기본값 15로 초기화
    # 이전 값과 비교하여 로그를 저장할 지 말 지 체크
    if floor != last_floor or arrow != last_arrow:  # 이전 값과 현재 값이 다를 때
      last_floor = floor
      last_arrow = arrow
      log_to_csv(floor, arrow)
      return
    else:  # 이전 값과 현재 값이 같을 때
      return


# 파악된 floor, arrow를 통해 로그 기록하는 함수
def log_to_csv(floor, arrow):
  # 클래스 ID를 클래스 이름으로 변경
  #{0: 'B2', 1: 'B1', 2: 'F1', 3: 'F2', 4: 'F3', 5: 'F4', 6: 'F5', 7: 'F6', 8: 'F7', 9: 'F8', 10: 'F9', 11: 'DOWN', 12: 'UP'}
  floor = class_name[int(floor)] if floor >= 0 else 'X'
  arrow = class_name[int(arrow)] if arrow >= 0 else 'X'
  with open(csv_path, 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([timestamp, floor, arrow])
  #print(floor, arrow)


# 메인 함수
def main():
  make_csv()
  while True:
    capture_image()

# 스크립트 실행
if __name__ == '__main__':
  main()

