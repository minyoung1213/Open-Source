from icrawler.builtin import GoogleImageCrawler
import os

# 이미지 저장 루트 디렉토리
# 이미지 저장 루트 디렉토리
root_dir = r'C:\Users\jiyunae\OneDrive\Desktop\Sookmyung\dataset\dumping'

# 크롤링 키워드 리스트
keywords = [
" body action dropping thing", "woman dropped thing", "picking something from the ground"
]

# 각 키워드마다 폴더 만들고 이미지 수집
for kw in keywords:
    folder_name = kw.replace(' ', '_')
    save_path = os.path.join(root_dir, folder_name)
    os.makedirs(save_path, exist_ok=True)

    print(f" [{kw}] → {save_path}")
    crawler = GoogleImageCrawler(storage={'root_dir': save_path})
    crawler.crawl(keyword=kw, max_num=100)
