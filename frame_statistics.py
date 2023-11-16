import copy

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
import onnxruntime as ort


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
    result = model_session.run(None, ort_inputs)[0][0][0]
    return result


def load_single_image(image: np.ndarray) -> np.ndarray:
    """
    Load image and perform preprocessing
    :param image_path: path to the test image
    :return: tensor suitable for cnn inference
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224)).astype(np.float32)
    image = np.expand_dims(image, 0)
    return image


def postprocess_prediction(raw_predition: float, threshold: float) -> int:
    """
    Convert float value to an int-encoded label with threshold
    :param raw_predition:
    :return: 1 if input is REAL, 0 otherwise
    """
    return 1 if (raw_predition > threshold) else 0


def _normalized_to_pixel_coordinates(
        normalized_x: float, normalized_y: float, image_width: int,
        image_height: int) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                          math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def visualize(
        image,
        detection_result
) -> np.ndarray:
    """Draws bounding boxes and keypoints on the input image and return it.
    Args:
      image: The input RGB image.
      detection_result: The list of all "Detection" entities to be visualize.
    Returns:
      Image with bounding boxes.
    """
    annotated_image = image.copy()
    height, width, _ = image.shape

    for detection in detection_result.detections:
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(annotated_image, start_point, end_point, (255, 0, 0), 3)
    return annotated_image


def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_iris_connections_style())

    return annotated_image


def plot_face_blendshapes_bar_graph(face_blendshapes):
    # Extract the face blendshapes category names and scores.
    face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
    face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
    # The blendshapes are ordered in decreasing score value.
    face_blendshapes_ranks = range(len(face_blendshapes_names))

    fig, ax = plt.subplots(figsize=(12, 12))
    bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
    ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
    ax.invert_yaxis()

    # Label each bar with values
    for score, patch in zip(face_blendshapes_scores, bar.patches):
        plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

    ax.set_xlabel('Score')
    ax.set_title("Face Blendshapes")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    base_options_face = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite')
    options_face = vision.FaceDetectorOptions(base_options=base_options_face)
    face_detector = vision.FaceDetector.create_from_options(options_face)

    sess = load_model('model.onnx')

    # cap = cv2.VideoCapture('IMG_3728.MOV')
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()

        resized = cv2.resize(frame, (500, 500))

        t = load_single_image(resized)
        pred = run_model(sess, t)
        formatted = postprocess_prediction(pred, 0.5)

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=resized)

        detection_result = detector.detect(image)
        face_detection_result = face_detector.detect(image)

        image_copy = np.copy(image.numpy_view())
        face_annotated_image = visualize(image_copy, face_detection_result)

        annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        det_img = copy.deepcopy(resized)

        font = cv2.FONT_HERSHEY_SIMPLEX
        if formatted:
            cv2.putText(det_img, f'{formatted}', (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.rectangle(det_img, (0, 0), (500, 500), color=(0, 255, 0), thickness=30)
        else:
            cv2.putText(det_img, f'{formatted}', (50, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.rectangle(det_img, (0, 0), (500, 500), color=(0, 0, 255), thickness=30)

        hor1 = cv2.hconcat([resized, annotated_image])
        hor2 = cv2.hconcat([det_img, face_annotated_image])
        cv2.imshow('frame', cv2.vconcat([hor1, hor2]))

        cv2.waitKey(1)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
