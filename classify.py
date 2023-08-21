import os
import cv2
from ultralytics import YOLO
from PIL import Image
from paddleocr import PaddleOCR, draw_ocr
import glob
import re
from collections import Counter
import torch


PATH = './cropped_images/'

img_paths = glob.glob(PATH+'*.jpg')

img_paths.sort()

# OCR 모델 초기화
ocr = PaddleOCR(lang='korean', cla=False)
a = []
b = []
for cnt,i in enumerate(img_paths):
    img = Image.open(i).convert('RGB')

    img_path = i
    results = ocr.ocr(img_path, cls = False)

    boxes = [temp[0] for temp in results[0]]
    texts = [temp[1][0] for temp in results[0]]
    scores = [temp[1][1] for temp in results[0]]

    a.append((texts,cnt))
    b.append(boxes)

# Load the YOLO model
model = YOLO('D:/best.pt')

# Predict on the input image
image_path = r"D:\real\realimage\KakaoTalk_20230813_160712289_01.jpg"
result = model.predict(image_path, save=False, conf=0.4, save_crop=True)

# Get detected bounding box data for the 'book_label' class
book_label_boxes = result[0].boxes.data[result[0].boxes.data[:, 5] == 2]  # Assuming 2 is the class ID for 'book_label'
book_label_reversed_boxes = result[0].boxes.data[result[0].boxes.data[:, 5] == 3]

# Open the original image
original_image = cv2.imread(image_path)

# Create a directory to save cropped images
os.makedirs('./cropped_images', exist_ok=True)

# Iterate through sorted bounding boxes and save cropped images
for i, box in enumerate(book_label_boxes):
    x1, y1, x2, y2, conf, class_id = box
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Crop the image
    cropped_image = original_image[y1:y2, x1:x2]

    # Save the cropped image
    save_path = f'./cropped_images/cropped_{i}.jpg'
    cv2.imwrite(save_path, cropped_image)

    print(f"Cropped image saved at: {save_path}")

# # 원 이미지지 보기
# cv2.namedWindow('happy',cv2.WINDOW_NORMAL)
# cv2.imshow('happy', original_image)
# if cv2.waitKey(0) & 0XFF==(29):
#     cv2.destroyAllWindows()
x_label = ['000','100','200','300','400','500','600','700''800','900']

a_filtered = []
a_combined = []
invalid = []
valid =[]
for entry in a:
    filtered_entry = [value for value in entry[0] if value not in x_label]
    a_filtered.append((filtered_entry, entry[1]))

for entry in a_filtered:
    combined_entry = ''.join(entry[0])
    a_combined.append((combined_entry, entry[1]))

reg = re.compile('\d{3}\D*\d*\D\d*\D.*')
book_valid_list = []
for i in a_combined:
    #OCR결과가 청구기호 형식을 갖춘 책 찾는 코드
    if reg.search(i[0]) :
        result = reg.search(i[0])
        foward, back = result.span()
        word = i[0][foward:back]
        book_valid_list.append((word,i[1]))
    # 그렇지 않은 책 찾는 코드
    if reg.search(i[0]) is None:
        invalid.append(i)

remove_dot = []
for i in book_valid_list:
    d = i[0].replace('.','')
    d = i[0].replace(':','')
    remove_dot.append((d,i[1]))

book_valid_list = remove_dot
        
# book_valid_list

book_sorted_list = sorted(book_valid_list)
# book_sorted_list


book_diff =[] ##책 분류기호 [000,100,200....900]
book_diff_list=[] ##기준서가가 아닌 곳에 잘못 분류된 책 위치
book_right_list=[] ##기준 서가에 제대로 분류된 책 



for i in book_sorted_list:
    book_diff.append((int(i[0][:3])//100*100)) ##분류기호 검출

count = Counter(book_diff)

if count:
    most_book_diff = count.most_common()[0][0] ##가장 많이 나온 책 분류기호를 기준으로 잡음

for i in book_sorted_list:
    if (int(i[0][:3])//100*100) != most_book_diff:
        book_diff_list.append(i)
        # if int(i[0][:3])//100*100 == 0 :
        #     print(i[1],  "000번 서가 위치로 이동하세요")
        # else:
        #     print(i[1], str(int(i[0][:3])//100*100)+"번 서가 위치로 이동하세요")
    else: 
        book_right_list.append(i)

## 텐서 dtype을 int로 변경 (해도 되고 안해도 됨)    

book_label_boxes = book_label_boxes.type(torch.int16)
book_label_reversed_boxes = book_label_reversed_boxes.type(torch.int16)


# try:
#     for i in invalid : ##식별 안되는 책(검정색)
#         original_image = cv2.rectangle(original_image, (int(book_label_boxes[i[1]][0]),int(book_label_boxes[i[1]][1])), (int(book_label_boxes[i[1]][2]), int(book_label_boxes[i[1]][3])), (0,0,0), 3)  
#         original_image = cv2.putText(original_image, "num"+str(cnt), (int(book_label_boxes[i[1]][0]),int(book_label_boxes[i[1]][1])), cv2.FONT_HERSHEY_COMPLEX,1 , (0,0,0), 2)
# except:
#     pass


try: 
    for cnt, i in enumerate(book_sorted_list,1): ##같은 서가 위치 다른책 위치변경(파란색)
        original_image = cv2.rectangle(original_image, (int(book_label_boxes[i[1]][0]),int(book_label_boxes[i[1]][1])), (int(book_label_boxes[i[1]][2]), int(book_label_boxes[i[1]][3])), (255,0,0), 3)
        original_image = cv2.putText(original_image, "num"+str(cnt), (int(book_label_boxes[i[1]][0]),int(book_label_boxes[i[1]][1])), cv2.FONT_HERSHEY_COMPLEX,1 , (255,0,0), 2)
except :
    pass

try:
    for j in book_diff_list: ##다른 서가에 위치한 책 알려줌(초록색)
        original_image = cv2.rectangle(original_image, (int(book_label_boxes[j[1]][0]),int(book_label_boxes[j[1]][1])), (int(book_label_boxes[j[1]][2]), int(book_label_boxes[j[1]][3])), (0,255,0), 3)
        if int(i[0][:3])//100*100 ==0 :
            original_image = cv2.putText(original_image, "move to 000th shelves", (int(book_label_boxes[j[1]][0]),int(book_label_boxes[j[1]][1])), cv2.FONT_HERSHEY_COMPLEX,1 , (0,255,0), 2)
        else:
            original_image = cv2.putText(original_image, "move to"+str(int(j[0][:3])//100*100)+"th shelves", (int(book_label_boxes[j[1]][0]),int(book_label_boxes[j[1]][1])), cv2.FONT_HERSHEY_COMPLEX,1 , (0,255,0), 2)
except:
    pass

try:
    for i in book_label_reversed_boxes: ##뒤집힌 책 (빨간색)
        original_image = cv2.rectangle(original_image, (int(i[0]),int(i[1])), (int(i[2]),int(i[3])), (0,0,255), 3)
        original_image = cv2.putText(original_image, "reversed" , (int(i[0]),int(i[1])), cv2.FONT_HERSHEY_COMPLEX,1 , (0,0,255), 2)
except:
    pass


original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
original_image = Image.fromarray(original_image)
original_image.save('book.jpg', "JPEG")