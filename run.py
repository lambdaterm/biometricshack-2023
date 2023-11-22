import os
import cv2
from landmarker import Landmarker
import mediapipe as mp
from face_detector import FaseDetector
from antispoofing import AntiSpoofing, AntiSpoofingCosine

DATA_DIR = 'data'


if __name__ == '__main__':

    landmarker = Landmarker('face_landmarker.task')
    face_detector = FaseDetector('blaze_face_short_range.tflite')

    # Модель обученная в рамках первого дня хакатона. Датасеты публичные - грязные не очищенные.
    # Рекомендуется делать центр-кроп в квадрат. Размер тензора 224 на 224.
    # Качество на тесте порядка 98%. Одинаково ошибается, как на риалах, так и на фейках.
    # Сеть не прунилась и не квантовалась.
    # Чувствительна к качеству камеры и освещению. При плохом освещении и плохом качестве - ошибается.
    # Классифицирует реальные изображения как фейки.
    # Требуется наличие лица в кадре. Рекомендуем использовать вместе с детектором лиц.

    model_224 = AntiSpoofing('models/model_224.onnx', 0.9249477386474609, (224, 224))
    print('Start model with small tensors testing ...')
    for i in os.listdir(DATA_DIR):
        image_from_file = cv2.imread(os.path.join(DATA_DIR, i))
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_from_file)
        detection_result = landmarker.detector.detect(image)
        if detection_result.face_landmarks:
            face_detection_results = True
        else:
            face_detection_results = False
        tens = model_224.load_single_image(os.path.join(DATA_DIR, i))
        prediction = model_224.run_model(tens)
        print(f'File name = {i}, Sigmoid = {prediction}, Face presence = {face_detection_results}, Is real face = {bool(model_224.postprocess_prediction(prediction))}')
    print('Done')

    # Модель с архитектурой аналогичной первой, но с большим размером тензора.
    # Датасеты публичные, но очищенные от мусора. Училась только на квадратных изображениях.
    # Настоятельно рекомендуем делать центр-кроп. На прямоугольных изображениях с соотношением
    # сторон большим чем 2 к 1 - результат не гарантируется.
    # Размер тензора 900 на 900.
    # Сеть не прунилась и не квантовалась. Работает существенно медленнее первой.
    # Чувствительна к качеству камеры и освещению.
    # Сильно меньше ошибается на реальных изображениях, но находит меньше фейков

    print('Start model with big tensors testing ...')
    model_900 = AntiSpoofing('models/model_900.onnx', 0.5, (900, 900))
    for i in os.listdir(DATA_DIR):
        image_from_file = cv2.imread(os.path.join(DATA_DIR, i))
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_from_file)
        detection_result = landmarker.detector.detect(image)
        if detection_result.face_landmarks:
            face_detection_results = True
        else:
            face_detection_results = False
        tens = model_900.load_single_image(os.path.join(DATA_DIR, i), is_center_crop=True)
        prediction = model_900.run_model(tens)
        print(f'File name = {i}, Sigmoid = {prediction}, Face presence = {face_detection_results}, Is real face = {bool(model_900.postprocess_prediction(prediction))}')
    print('Done')

    # Модель простейшего классификатора на три класса. Использовали metric-learning и косинусное расстояние
    # для растягивания классов. Есть негативный класс - отсутствие лица в кадре. Классификационный слой можно убрать
    # и использовать набор эталонных лиц с конкретной камеры в качестве центров классов.
    #
    # Датасеты публичные, грязные.
    # Размер тензора 224 на 224.
    # Сеть не прунилась и не квантовалась. Работает быстро.
    # Почти нет ошибок на классе реальных. Находит меньше всего фейков. Не требует детектора лиц,
    # так как есть негативный класс. Не требует центрального кропа, но не училась на прямоугольных изображениях.

    print('Start model with cosine tensors testing ...')
    model_cosine = AntiSpoofingCosine('models/model_cosine_224.onnx', (224, 224))
    for i in os.listdir(DATA_DIR):
        tens = model_cosine.load_single_image(os.path.join(DATA_DIR, i))
        prediction = model_cosine.run_model(tens)
        print(
            f'File name = {i}, Class confidences = {prediction}, Is real face = {bool(model_cosine.postprocess_prediction(prediction))}')
    print('Done')

    # Выводы и рекомендации:
    # Для бизнес задач рекомендуем применять все три сети вместе как ансамбль сетей.
    # Подбирать значения трешхолдов сетей под конкретную камеру.
    # Делать детекцию лица и применять центр-кроп по лицу.
    # Лучшее качество у нас на тесте показала модель - model_900 с детектором лиц и центр-кропом.
    # Ансамбль из трех моделей не валит реальные лица вообще, но находит меньше всего фейков.
    # В качестве доработки решения планируем:
    # 1. Очистить публичные тренировочные датасеты от ошибок. Нормализовать изображения.
    # Сформировать собственный датасет, разделив исходные по типам спуфинга.
    # 2. Обучить модели под каждый тип спуфинга отдельно.
    # 3. Использовать ембединг лица при обучении. Подавать на вход сети не только тензор с исходным изображением, но и
    # ембединг лица с него, полученный с хорошей публичной сети. Варианты InsightFace, DeepFace, ArcFace, MediaPipe etc
    # 4. Использовать модель возвращающую карту глубины и подавать ее вместе с исходным изображением.
    # 5. Алгоритмически получать белый шум матрицы камеры и подавать его вместе с исходным тензором. Требует отсутствия
    # кодирования изображений с потерями. Кодирование в Jpg портит шумы матрицы.
    # 6. Доработать идею с metric learning и косинусным расстоянием. Это позволит сделать решение более гибким, и
    # настраивать его под новые типы спуфинга без переучивания моделей.
    #
    # Основной проблемой было отсутствие нормальных чистых и хороших публичных датасетов.




