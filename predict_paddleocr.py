from ultralytics import YOLO
import cv2
from PIL import Image
from paddleocr import PaddleOCR

modelpath = r"D:\best.pt" ##model path


class image_predict:
    def __init__(self, imagepath):
        
        self.imagepath = imagepath
        self.model = YOLO(modelpath)
        self.ocr = PaddleOCR(lang="korean")
        #self.names = {0: 'book', 1: 'book_reversed', 2: 'book_label', 3: 'book_label_reversed'}


    def show_predict(self): ## 모델 예측 사진 보여주는 함수
        wordlist= []
        reversed_wordlist=[]
        img = cv2.imread(self.imagepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.model.predict(img)
        plots = result[0].plot()


        img = Image.fromarray(plots)
        img.save('predict_image.jpg','JPEG')

        return img

    def ocr_list(self): ## OCR 반환 하는 함수
        img = cv2.imread(self.imagepath)
        result = self.model.predict(self.imagepath)

        reversed_book_count = 0
        valid_book_count = 0

        wordlist = []
        reversedlist = []
        for cnt, i in enumerate(result[0].boxes.boxes):
            if i[-1] == 3:
                reversed_book_count += 1
                x1,y1 = int(result[0].boxes.boxes[cnt][0]),int(result[0].boxes.boxes[cnt][1])
                x2,y2 = int(result[0].boxes.boxes[cnt][2]),int(result[0].boxes.boxes[cnt][3])

                crop_img = img[y1:y2, x1:x2]
                rotate_img = cv2.rotate(crop_img, cv2.ROTATE_180)
                results = self.ocr.ocr(rotate_img, cls= False)
                
                reversed_words=[]
                for idx, i in enumerate(results): ###
                    for j in range(len(i)):
                        reversed_words.append(i[j][1][0])
                    new_reversed_words = ''.join(reversed_words)
                    print(new_reversed_words)
                    reversedlist.append(new_reversed_words)

            elif i[-1] == 2:
                valid_book_count +=1
                x1,y1 = int(result[0].boxes.boxes[cnt][0]),int(result[0].boxes.boxes[cnt][1])
                x2,y2 = int(result[0].boxes.boxes[cnt][2]),int(result[0].boxes.boxes[cnt][3])

                crop_img = img[y1:y2, x1:x2]
                results = ocr.ocr(crop_img, cls= False)

                valid_words=[]
                for idx, i in enumerate(results): ###
                    for j in range(len(i)):
                        valid_words.append(i[j][1][0])
                        print(i[j][1][0])
                    new_words = ''.join(valid_words)
                    print(new_words)
                    wordlist.append(new_words)

        return wordlist, reversedlist
    
    def reversed_book_location(self): ## 뒤집힌 책 사진 보여주는 함수
        wordlist= []
        reversed_wordlist=[]
        img = cv2.imread(self.imagepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.model.predict(self.imagepath)
        for cnt, i in enumerate(result[0].boxes.boxes):
            if i[-1] == 2:
                x1,y1 = int(result[0].boxes.boxes[cnt][0]),int(result[0].boxes.boxes[cnt][1])
                x2,y2 = int(result[0].boxes.boxes[cnt][2]),int(result[0].boxes.boxes[cnt][3])
                
                img = cv2.rectangle(img, (x1,y1),(x2,y2),(255,0,0),3,3)

        img = Image.fromarray(img)
        img.save('reversed_book_location_image.jpg','JPEG')

        return img
    
    
    def valid_book_location(self): ##올바른 책 사진 보여주는 함수
        wordlist= [] 
        reversed_wordlist=[]
        img = cv2.imread(self.imagepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.model.predict(self.imagepath)
        for cnt, i in enumerate(result[0].boxes.boxes):
            if i[-1] == 2:
                x1,y1 = int(result[0].boxes.boxes[cnt][0]),int(result[0].boxes.boxes[cnt][1])
                x2,y2 = int(result[0].boxes.boxes[cnt][2]),int(result[0].boxes.boxes[cnt][3])
                
                img = cv2.rectangle(img, (x1,y1),(x2,y2),(0,0,255),3,3)

        img = Image.fromarray(img)
        img.save('valid_book_location.jpg', "JPEG")

        return img
