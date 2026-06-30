import cv2
from ultralytics import YOLO

# 1. AI 모델과 테스트 이미지 불러오기
model = YOLO('yolo11n.pt')
img_path = 'test.jpg' # 아까 사용한 사진 이름
img = cv2.imread(img_path)

# 2. 이미지 분석 (객체 찾기)
results = model(img)

# 3. 찾은 객체들에 대해 하나씩 확인하기
for r in results:
    boxes = r.boxes
    for box in boxes:
        # 네모 박스 좌표 (왼쪽 위 x, y / 오른쪽 아래 x, y)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # 객체의 번호(ID) 확인 (YOLO에서 0번은 '사람'입니다)
        class_id = int(box.cls[0])
        
        # 4. 만약 '사람(0번)'이 아니라면 블러 처리하기!
        if class_id != 0:
            # 박스 영역만큼 잘라내기
            roi = img[y1:y2, x1:x2]
            
            # 모자이크(블러) 강하게 적용 (50, 50 숫자가 클수록 흐려짐)
            blur_roi = cv2.blur(roi, (50, 50))
            
            # 원본 이미지에 흐려진 부분 덮어쓰기
            img[y1:y2, x1:x2] = blur_roi

# 5. 완성된 사진을 새 이름으로 저장하기
cv2.imwrite('blur_result.jpg', img)
print("블러 처리 완료! 폴더 안에 blur_result.jpg를 확인하세요!")