import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

class ObjectDetector:
    def __init__(self):
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]
        self.LABEL_COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))
        self.prototxt_path = os.getcwd() + '\\MobileNetSSD_deploy.prototxt.txt'
        self.model_path = os.getcwd() + '\\MobileNetSSD_deploy.caffemodel'
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt_path, self.model_path)
        self.save_path = os.getcwd() + '\\data\\Detected_Img' #결과 이미지 저장 경로 고정

    def perform_object_detection(self, image_path):
        detections_list = [] # 최종 결과 객체 list로 return => C# 에서 가공해서 뽑아야됨

        cv2_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        self._show_image('original image', cv2_image)

        (h, w) = cv2_image.shape[:2]
        resized = cv2.resize(cv2_image, (300, 300))
        blob = cv2.dnn.blobFromImage(resized, 0.007843, (300, 300), 127.5)

        self.net.setInput(blob)
        detections = self.net.forward()

        conf = 0.2
        vis = cv2_image.copy()

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > conf:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                print("[INFO] {} : [ {:.2f} % ]".format(self.CLASSES[idx], confidence * 100))

                cv2.rectangle(vis, (startX, startY), (endX, endY), self.LABEL_COLORS[idx], 1)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(vis, "{} : {:.2f}%".format(self.CLASSES[idx], confidence * 100), (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, self.LABEL_COLORS[idx], 2)

                detections_list.append({
                    'class': self.CLASSES[idx],
                    'confidence': confidence,
                    'box': (startX, startY, endX, endY)
                })

        self._show_image('Object Detection', vis, figsize=(16, 10)) # 모든 객체에 대한 검출 결과를 출력

        return detections_list

    def _show_image(self, title, img, figsize=(8, 5)):
        plt.figure(figsize=figsize)

        if type(img) == list:
            if type(title) == list:
                titles = title
            else:
                titles = []

                for i in range(len(img)):
                    titles.append(title)

            for i in range(len(img)):
                if len(img[i].shape) <= 2:
                    rgbImg = cv2.cvtColor(img[i], cv2.COLOR_GRAY2RGB)
                else:
                    rgbImg = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)

                plt.subplot(1, len(img), i + 1), plt.imshow(rgbImg)
                plt.title(titles[i])
                plt.xticks([]), plt.yticks([])

            plt.show()
        else:
            if len(img.shape) < 3:
                rgbImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            plt.imshow(rgbImg)
            plt.title(title)
            plt.xticks([]), plt.yticks([])
            plt.show()

    def save_images(self, detections, input_image_path):
        #detective 객체 따로 저장 되는것이 아닌 한 이미지에 객체 다 담겨서 저장 되도록
        if self.save_path is None:
            raise ValueError("Save path is not specified.")

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        cv2_image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)

        for detection in detections:
            class_name = detection['class']
            confidence = detection['confidence']
            box = detection['box']
            (startX, startY, endX, endY) = box

            cv2.rectangle(cv2_image, (startX, startY), (endX, endY), (0, 255, 0), 1)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(cv2_image, "{} : {:.2f}%".format(class_name, confidence * 100),
                        (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        #파일명은 Original 이미지 이름 가져와 +detected 로  저장
        original_image_name = os.path.splitext(os.path.basename(input_image_path))[0]
        filename = f"{original_image_name}_detected.jpg"
        save_image_path = os.path.join(self.save_path, filename)
        cv2.imwrite(save_image_path, cv2_image)

        print(f"Saved: {save_image_path}")

        return filename