import cv2
import mediapipe as mp
import os
import csv

# 설정
dataset_dir = r'C:\dataset'  # 이미지 폴더 루트
output_csv = "pose_data1.csv"  # 결과 저장 파일

# MediaPipe pose 모델
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# CSV 파일 열기
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)

    # 헤더 작성
    header = ["label"]
    for i in range(33):  # 33개의 관절
        header += [f"x{i}", f"y{i}", f"z{i}"]
    writer.writerow(header)

    # 폴더 순회
    for label in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, label)
        if not os.path.isdir(class_dir):
            continue

        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            image = cv2.imread(img_path)
            if image is None:
                continue

            # MediaPipe 처리
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if not results.pose_landmarks:
                continue  # 포즈 감지 실패 시 건너뜀

            # 좌표 추출
            row = [label]
            for lm in results.pose_landmarks.landmark:
                row += [lm.x, lm.y, lm.z]  # visibility는 생략 가능
            writer.writerow(row)

print(f"CSV 저장 완료: {output_csv}")
