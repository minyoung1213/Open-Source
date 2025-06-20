import os

# 리네이밍할 이미지가 있는 폴더 경로
folder_path = r'C:\Users\jiyunae\OneDrive\Desktop\Sookmyung\dataset\normal'

# 확장자 필터 (필요시 jpg/png 변경 가능)
valid_exts = ['.jpg', '.jpeg', '.png']

# 폴더 내 파일 정렬
files = sorted([f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in valid_exts])

# 리네이밍 실행
for i, filename in enumerate(files, 1):
    ext = os.path.splitext(filename)[1].lower()
    new_name = f"{i}{ext}"
    src = os.path.join(folder_path, filename)
    dst = os.path.join(folder_path, new_name)
    os.rename(src, dst)

print("리네이밍 완료!!")
