import copy
import time
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Tuple, Union

from antispoofing import AntiSpoofing, AntiSpoofingCosine
from landmarker import Landmarker
from face_detector import FaseDetector
from typing import Tuple

ModelResult = Tuple[int, str, float]


def postprocess_prediction(raw_predition: float, threshold: float) -> ModelResult:
    """
    Convert float value to an int-encoded label with threshold
    :param raw_predition:
    :return: 1 if input is REAL, 0 otherwise
    """
    return (1, 'Real', raw_predition) if (raw_predition > threshold) else (0, 'Fake', raw_predition)



if __name__ == '__main__':

    ###
    print('Face detector initialization ...')
    face_detector_init_time = time.time()
    landmarker = Landmarker('face_landmarker.task')
    face_detector = FaseDetector('blaze_face_short_range.tflite')
    print(f'Detector initialized in {round(time.time() - face_detector_init_time, 4)} sec')
    ###
    print('Models for Spoofing detection initialization ...')
    model_init_time = time.time()
    model_small = AntiSpoofing('models/model_224.onnx', 0.1, (224, 224))
    model_big = AntiSpoofing('models/model_900.onnx', 0.2, (900, 900))
    model_dolls = AntiSpoofing('models/model_doll_224.onnx', 0.98, (224, 224))
    model_cosine = AntiSpoofingCosine(model_path='models/model_cosine_224.onnx', image_size=(224, 224))
    print(f'Models initialized in  {round(time.time() - model_init_time, 4)} sec')
    ###
    # Забираем видео из видео потока
    # cap = cv2.VideoCapture('IMG_3728.MOV')
    # Забираем видео с мобильного телефона
    # cap = cv2.VideoCapture('http://172.16.74.154:8080/video', cv2.CAP_ANY)
    # cap.set(3, 900)
    # cap.set(4, 900)
    # Забираем видео с web камеры
    web_cam_screen_size = (800, 600)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, web_cam_screen_size[0])
    cap.set(4, web_cam_screen_size[0])
    ###
    frames = 0
    fps = 0
    frame_time = time.time()
    frame_time_in_sec = 0
    while cap.isOpened():

        ###
        frame_time = time.time() - frame_time
        frame_time_in_sec = frame_time_in_sec + frame_time
        frames += 1
        if frame_time_in_sec > 1:
            frame_time_in_sec = 0
            fps = frames
            frames = 0
        if fps != 0:
            print(f'fps = {fps} Frame performing time = {str(int(frame_time * 1000))} ms')
        frame_time = time.time()

        ###
        ret, frame = cap.read()
        original_frame = frame.copy()
        print(f'Frame_size = {frame.shape}')
        # frame = cv2.resize(frame, (900, 900))
        annotated_image = frame

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = landmarker.detector.detect(image)
        # print_color is in BGR notation
        if detection_result.face_landmarks:

            ###
            tensor = model_small.load_single_image(original_frame, is_center_crop=True)
            result = model_small.run_model(tensor)
            post_processed_result = postprocess_prediction(result, model_small.threshold)
            print(('Small:', post_processed_result))
            ###
            tensor = model_big.load_single_image(original_frame, is_center_crop=True)
            result = model_big.run_model(tensor)
            post_processed_result_for_big_tensor = postprocess_prediction(result, model_big.threshold)
            print(('Big:  ', post_processed_result_for_big_tensor))
            ###
            tensor = model_cosine.load_single_image(original_frame, is_center_crop=True)
            result = model_cosine.run_model(tensor)
            result_for_cosine, result_for_cosine_as_text = model_cosine.postprocess_prediction_as_text(result)
            print(('Cosine:  ', result))
            ###
            tensor = model_dolls.load_single_image(original_frame, is_center_crop=True)
            result = model_dolls.run_model(tensor)
            post_processed_result_for_dolls = postprocess_prediction(result, model_dolls.threshold)
            print(('Doll:  ', post_processed_result_for_dolls))
            ###

            if post_processed_result[0] and post_processed_result_for_big_tensor[0] and result_for_cosine:
                print_color = (0, 255, 0)
            elif post_processed_result[0] or post_processed_result_for_big_tensor[0] or result_for_cosine:
                print_color = (0, 255, 255)
            else:
                print_color = (0, 0, 255)

            cv2.putText(frame, f'Small  = {post_processed_result[1]}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, print_color, 2, cv2.LINE_AA)
            cv2.putText(frame, f'Big     = {post_processed_result_for_big_tensor[1]}', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, print_color, 2,
                        cv2.LINE_AA)
            cv2.putText(frame, f'Cosine = {result_for_cosine_as_text}', (50, 110), cv2.FONT_HERSHEY_SIMPLEX,
                        1, print_color, 2,
                        cv2.LINE_AA)
            cv2.putText(frame, f'Doll    = {post_processed_result_for_dolls[1]}', (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, print_color, 2,
                        cv2.LINE_AA)

            cv2.rectangle(frame, (0, 0), web_cam_screen_size, color=print_color, thickness=30)
            face_detection_result_bool = 0

            face_detection_result = face_detector.face_detector.detect(image)
            image_copy = np.copy(image.numpy_view())
            annotated_image = landmarker.draw_landmarks_on_image(image.numpy_view(), detection_result)

            if post_processed_result[0]:
                print(f'Face is {post_processed_result[1]} with confidence {post_processed_result[2]}')
            else:
                print(f'Face is {post_processed_result[1]} with confidence {1-post_processed_result[2]}')

        horizontal_line = cv2.hconcat([annotated_image, frame])
        cv2.imshow("Camera", horizontal_line)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()
