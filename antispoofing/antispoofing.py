import onnxruntime as ort
import cv2
import numpy as np
import albumentations as A

from typing import Tuple, Union


class Antispoofing:
    def __init__(self, model_path: str, threshold: float, image_size: Tuple[int, int]):
        self.model_path = model_path
        self.threshold = threshold
        self.image_size = image_size
        self.sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    def run_model(self, input_tensor: np.ndarray) -> float:
        ort_inputs = {self.sess.get_inputs()[0].name: input_tensor}
        result = self.sess.run(None, ort_inputs)[0][0][0]
        return result

    def load_single_image(self, image: Union[str, np.ndarray], is_center_crop=False) -> np.ndarray:
        if isinstance(image, str):
            image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if is_center_crop:
            image = self.center_crop(image)
        image = cv2.resize(image, self.image_size).astype(np.float32)
        image = np.expand_dims(image, 0)
        return image

    def postprocess_prediction(self, raw_predition: float) -> int:
        return 1 if (raw_predition > self.threshold) else 0

    def center_crop(self, image: np.ndarray) -> np.ndarray:
        transform = A.Compose([
            A.CenterCrop(height=900, width=900, always_apply=True)
        ])
        return transform(image=image)['image']
