import cv2
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt



def process_video(input_video_path):
    cap = cv2.VideoCapture(input_video_path)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        i += 1
        resized_frame = frame
        # print(i)
        if i == 850:
            cv2.imwrite('frame_850.jpg', resized_frame)
        

    cap.release()

process_video('output_2.mp4')