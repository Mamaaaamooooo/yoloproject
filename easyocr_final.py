from ultralytics import YOLO
import cv2
from easyocr import Reader
from PIL import Image
from glob import glob
import re
from collections import Counter
import os
import datetime as dt

imagepath = r"C:\Users\user\Desktop\testtest\KakaoTalk_20230819_153354605.jpg"

model = YOLO(r"D:\best.pt")

path = 'd:/cropped_images/{}/'.format(dt.datetime.now().strftime(("%Y%m%d%H%M%S")))
os.makedirs(path, exist_ok=True)

img = cv2.imread(imagepath)
result = model.predict(img, conf=0.05)
count = 1
book_label_boxes = []
book_label_reversed_boxes = []
for cnt,i in enumerate(result[0].boxes.data):
    x1,y1 = int(result[0].boxes.data[cnt][0]),int(result[0].boxes.data[cnt][1])
    x2,y2 = int(result[0].boxes.data[cnt][2]),int(result[0].boxes.data[cnt][3])
    if i[-1] == 2:
        crop_img = img[y1:y2, x1:x2]
        cv2.imwrite(path+str(count).zfill(3)+'.jpg', crop_img)
        count+=1
        book_label_boxes.append([x1,y1,x2,y2])
    if i[-1] ==3:
        book_label_reversed_boxes.append([x1,y1,x2,y2])

PATH = path

img_paths = glob(PATH+'*.jpg')

##OCR모델 초기화
reader = Reader(['ko'])
a = []
a_filtered = []
a_combined = []
invalid = []
valid =[]

for i in img_paths:
    result = reader.readtext(i,detail=0)
    result = ''.join(result)
    a.append(result)

x_label = ['000','100','200','300','400','500','600','700''800','900']

remove_dot = []
for cnt, i in enumerate(a) :
    i = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]','', i)
    i = re.sub(' ', '', i)

    remove_dot.append((i,cnt))

a_combined = remove_dot




reg = re.compile('\d{3}\D*\d*\D\d*\D.*')
book_valid_list = []
invalid = []
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

unrecog_dict = []
for i in invalid:
    
    vid = re.compile('\d+\D+')
    if vid.search(i[0]):
        cmd = vid.search(i[0])
        foward, back = cmd.span()
        numbers = i[0][foward:back]
        sentence = i[0][back:]
        unrecog_dict.append((numbers,sentence,i[1]))



# book_valid_list

new_dict = []
for i in book_valid_list:
    vid = re.compile('\d*')
    cmd = vid.search(i[0])
    foward, back = cmd.span()
    numbers = i[0][foward:back]
    sentence = i[0][back:]
    new_dict.append((numbers,sentence,i[1]))


book_sorted_list = sorted(new_dict, key=lambda x: (x[0], x[1]))


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

# book_label_boxes = book_label_boxes.type(torch.int16)
# book_label_reversed_boxes = book_label_reversed_boxes.type(torch.int16)


# try:
#     for i in invalid : ##식별 안되는 책(검정색)
#         original_image = cv2.rectangle(original_image, (int(book_label_boxes[i[1]][0]),int(book_label_boxes[i[1]][1])), (int(book_label_boxes[i[1]][2]), int(book_label_boxes[i[1]][3])), (0,0,0), 3)  
#         original_image = cv2.putText(original_image, "num"+str(cnt), (int(book_label_boxes[i[1]][0]),int(book_label_boxes[i[1]][1])), cv2.FONT_HERSHEY_COMPLEX,1 , (0,0,0), 2)
# except:
#     pass
original_image = cv2.imread(imagepath)

if unrecog_dict:
    for i in unrecog_dict : ##식별 안되는 책(검정색)
        original_image = cv2.rectangle(original_image, (int(book_label_boxes[i[2]][0]),int(book_label_boxes[i[2]][1])), (int(book_label_boxes[i[2]][2]), int(book_label_boxes[i[2]][3])), (0,0,0), 2)
        original_image = cv2.putText(original_image, "unrecog", (int(book_label_boxes[i[2]][0]),int(book_label_boxes[i[2]][1])), cv2.FONT_HERSHEY_COMPLEX,1 , (0,0,0), 2)


for cnt, i in enumerate(book_sorted_list,1): ##같은 서가 위치 다른책 위치변경(파란색)

    original_image = cv2.rectangle(original_image, (int(book_label_boxes[i[2]][0]),int(book_label_boxes[i[2]][1])), (int(book_label_boxes[i[2]][2]), int(book_label_boxes[i[2]][3])), (255,0,0), 2)
    original_image = cv2.putText(original_image, "num"+str(cnt), (int(book_label_boxes[i[2]][0]),int(book_label_boxes[i[2]][1])), cv2.FONT_HERSHEY_COMPLEX,1 , (255,0,0), 2)
    print(cnt)

try:
    for j in book_diff_list: ##다른 서가에 위치한 책 알려줌(초록색)
        original_image = cv2.rectangle(original_image, (int(book_label_boxes[j[2]][0]),int(book_label_boxes[j[2]][1])+20), (int(book_label_boxes[j[2]][2]), int(book_label_boxes[j[2]][3])-20), (0,255,0), 3)
        if int(i[0][:3])//100*100 ==0 :
            original_image = cv2.putText(original_image, "move to 000th", (int(book_label_boxes[j[2]][0]+3),int(book_label_boxes[j[2]][1]+3)), cv2.FONT_HERSHEY_COMPLEX,1 , (0,255,0), 1)
        else:
            original_image = cv2.putText(original_image, "move to "+str(int(j[0][:3])//100*100), (int(book_label_boxes[j[2]][0]+3),int(book_label_boxes[j[2]][1]+3)), cv2.FONT_HERSHEY_COMPLEX_SMALL,1 , (0,255,0), 1)
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
original_image.save('d:/data2/result/book1.jpg', "JPEG")
original_image
