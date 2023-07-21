#Main Test 쉽게 필요에 의해 주석 풀고 사용 하면 됨.

#Main v1. ----이 부분은 내부적으로 test 할 때 -----
from ObjectDetection import ObjectDetector
import os
import sys
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Image file path is missing.")
        sys.exit(1)

    input_image_path = sys.argv[1]

    local_path = os.getcwd()
    #input_image_path = local_path + '\\data\\Original_Img\\CatDog1.jpg'
    save_path = local_path + '\\data\\Detected_Img'

    detector = ObjectDetector()

    detections = detector.perform_object_detection(input_image_path)

    for detection in detections:
         print(detection)

    # Save detected images
    detector.save_images(detections, input_image_path)

#main v2 C# 용
#C# 에서 ProcessStartInfo를 사용하여 파이썬 스크립트를 실행하고, 이미지 파일 경로를 인자로 전달할 때,
# 파이썬 스크립트가 sys.argv[1]을 통해 이미지 경로를 읽고 해당 경로에 따라 객체 검출을 수행도록..
# import sys
# from ObjectDetection import ObjectDetector
# import os
# if __name__ == "__main__":
#     # 파이썬 스크립트에 전달된 인자 읽기
#     if len(sys.argv) < 2:
#         print("Error: Image file path is missing.")
#         sys.exit(1)
#
#     input_image_path = sys.argv[1]
#
#     # 결과 이미지 저장 경로
#     save_path = os.getcwd() + '\\data\\Detected_Img'
#
#     detector = ObjectDetector()
#
#     detections = detector.perform_object_detection(input_image_path)
#
#     for detection in detections:
#          print(detection)
#
#     # Save detected images
#     detector.save_images(detections, input_image_path)