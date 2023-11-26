import cv2
import torch

# 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

# HTTP 스트림 URL
stream_url = 'http://172.16.100.155:8080/video'

# 비디오 스트림을 시작합니다.
cap = cv2.VideoCapture(stream_url)

# 스트림이 올바르게 열렸는지 확인합니다.
if not cap.isOpened():
    print("비디오 스트림을 열 수 없습니다.")
else:
    while True:
        # 현재 프레임을 읽어옵니다.
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다. 스트리밍을 종료합니다.")
            break

        # YOLO 모델을 사용하여 프레임에서 객체를 검출합니다.
        results = model(frame)

        results.render()

        # 위 코드를 아래와 같이 수정합니다.
        for img in results.imgs:
            cv2.imshow('YOLOv5 Detection', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q'를 누르면 스트리밍을 중지합니다.
                break

# 비디오 스트림을 종료합니다.
cap.release()
cv2.destroyAllWindows()
