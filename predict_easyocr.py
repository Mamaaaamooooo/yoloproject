from ultralytics import YOLO
import cv2
from easyocr import Reader
from PIL import Image

modelpath = r"D:\best.pt" ##model path


class image_predict:
    def __init__(self, imagepath): ##모델 예측 사진 보여주는 함수
        
        self.imagepath = imagepath
        self.model = YOLO(modelpath)
        self.reader = Reader(['ko'])
        #self.names = {0: 'book', 1: 'book_reversed', 2: 'book_label', 3: 'book_label_reversed'}


    def show_predict(self): ## 
        wordlist= []
        reversed_wordlist=[]
        img = cv2.imread(self.imagepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.model.predict(img, show_conf=False, conf = 0.5)
        plots = result[0].plot()

        img = Image.fromarray(plots)
        img.save('predict_image.jpg','JPEG')

        return img


    def ocr_list(self): ## OCR 검출 함수
        
        count = 1
        wordlist= []
        reversed_wordlist=[]
        img = cv2.imread(self.imagepath)
        result = self.model.predict(self.imagepath)
        for cnt, i in enumerate(result[0].boxes.boxes):
            if i[-1] == 3:
                x1,y1 = int(result[0].boxes.boxes[cnt][0]),int(result[0].boxes.boxes[cnt][1])
                x2,y2 = int(result[0].boxes.boxes[cnt][2]),int(result[0].boxes.boxes[cnt][3])

                crop_img = img[y1:y2, x1:x2]
                rotate_img = cv2.rotate(crop_img, cv2.ROTATE_180)
                results = self.reader.readtext(rotate_img, detail = 0)
                reversed_wordlist.append(''.join(results))
            elif i[-1] == 2:
                x1,y1 = int(result[0].boxes.boxes[cnt][0]),int(result[0].boxes.boxes[cnt][1])
                x2,y2 = int(result[0].boxes.boxes[cnt][2]),int(result[0].boxes.boxes[cnt][3])

                crop_img = img[y1:y2, x1:x2]
                results = self.reader.readtext(crop_img, detail = 0)
                wordlist.append(''.join(results))


        return wordlist, reversed_wordlist
    
    def reversed_book_location(self): ##뒤집힌 책 보여주는 함수
        wordlist= []
        reversed_wordlist=[]
        img = cv2.imread(self.imagepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.model.predict(self.imagepath, conf=0.5)
        for cnt, i in enumerate(result[0].boxes.boxes):
            if i[-1] == 3:
                x1,y1 = int(result[0].boxes.boxes[cnt][0]),int(result[0].boxes.boxes[cnt][1])
                x2,y2 = int(result[0].boxes.boxes[cnt][2]),int(result[0].boxes.boxes[cnt][3])
                
                img = cv2.rectangle(img, (x1,y1),(x2,y2),(255,0,0),3,3)

        img = Image.fromarray(img)
        img.save('reversed_book_location_image.jpg','JPEG')

        return img
        
    def valid_book_location(self): ##바른 책 보여주는 함수
        wordlist= []
        reversed_wordlist=[]
        img = cv2.imread(self.imagepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.model.predict(self.imagepath, conf=0.5)
        for cnt, i in enumerate(result[0].boxes.boxes):
            if i[-1] == 2:
                x1,y1 = int(result[0].boxes.boxes[cnt][0]),int(result[0].boxes.boxes[cnt][1])
                x2,y2 = int(result[0].boxes.boxes[cnt][2]),int(result[0].boxes.boxes[cnt][3])
                
                img = cv2.rectangle(img, (x1,y1),(x2,y2),(0,0,255),3,3)

        img = Image.fromarray(img)
        img.save('valid_book_location.jpg', "JPEG")

        return img
            