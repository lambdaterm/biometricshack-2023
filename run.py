import numpy as np

import cv2
import onnxruntime as ort
import os


DATA_DIR = 'data'


def load_model(model_path: str) -> ort.InferenceSession:
    """
    Load ONNX model from model_path
    :param model_path: path to the model
    :return: InferenceSession
    """
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    return session


def run_model(model_session: ort.InferenceSession, input_tensor: np.ndarray) -> float:
    """
    Predict result for input tensor
    :param model_session: InferenceSession instance
    :param input_tensor: input tensor
    :return: a value [0; 1] that shows liveness representation
    """
    ort_inputs = {model_session.get_inputs()[0].name: input_tensor}
    result = sess.run(None, ort_inputs)[0][0][0]
    return result


def load_single_image(image_path: str, size: tuple) -> np.ndarray:
    """
    Load image and perform preprocessing
    :param size:
    :param image_path: path to the test image
    :return: tensor suitable for cnn inference
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, size).astype(np.float32)
    image = np.expand_dims(image, 0)
    return image


def postprocess_prediction(raw_predition: float, threshold: float) -> int:
    """
    Convert float value to an int-encoded label with threshold
    :param raw_predition:
    :return: 1 if input is REAL, 0 otherwise
    """
    return 1 if (raw_predition > threshold) else 0


if __name__ == '__main__':
    print('Start model with small tensors testing ...')
    sess = load_model('models/model_224.onnx')
    for i in os.listdir(DATA_DIR):
        tens = load_single_image(os.path.join(DATA_DIR, i), (224, 224))
        prediction = run_model(sess, tens)
        print(f'{i}, {prediction}, {postprocess_prediction(prediction, 0.9249477386474609)}')
    print('Done')

    print('Start model with big tensors testing ...')
    sess = load_model('models/model_900.onnx')
    for i in os.listdir(DATA_DIR):
        tens = load_single_image(os.path.join(DATA_DIR, i), (900, 900))
        prediction = run_model(sess, tens)
        print(f'{i}, {prediction}, {postprocess_prediction(prediction, 0.5)}')
    print('Done')