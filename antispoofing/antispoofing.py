import onnxruntime as ort
import cv2
import numpy as np
import albumentations as A
# from typing import List
from typing import Tuple, Union, List



class BaseONNXModel:

    def __init__(self, **kwargs):
        assert kwargs.get('model_path', False), "Need path to model"
        self.model_path = kwargs['model_path']
        self.image_size = kwargs['image_size'] if kwargs.get('image_size') else (224,224)
        self.sess = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])


    def run_model(self, input_tensor: np.ndarray) -> float:
        ort_inputs = {self.sess.get_inputs()[0].name: input_tensor}
        result = self.sess.run(None, ort_inputs)[0][0]
        if len(result) == 1:
            return result[0]
        else:
            return result

    def load_single_image(self, image: Union[str, np.ndarray], is_center_crop=False) -> np.ndarray:
        pass

    def postprocess_prediction(self, raw_prediction: Union[float, np.ndarray]) -> int:
        pass

    def center_crop_without_albumentations(self, image: np.ndarray, box: Union[List[int], Tuple[int, ...]]) -> np.ndarray:
        box = list(map(int, box))

        x1, y1, x2, y2 = box[:4]
        h, w = image.shape[:2]

        # Restrict coordinates to canvas
        x1 = x1 if x1 > 0 else 0
        y1 = y1 if y1 > 0 else 0

        x2 = x2 if x2 < w else w
        y2 = y2 if y2 < h else h

        out = image[y1:y2, x1:x2]
        return out

    def center_crop_wa(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if h > w:
            offset = (h - w) // 2
            box = [0, offset, w, w + offset]
        else:
            offset = (w - h) // 2
            box = [offset, 0, h + offset, h]

        return self.center_crop_without_albumentations(image, box)


class AntiSpoofing(BaseONNXModel):
    def __init__(self, model_path: str, threshold: float, image_size: Tuple[int, int]):
        super().__init__(model_path=model_path, image_size=image_size)
        self.threshold = threshold

    def load_single_image(self, image: Union[str, np.ndarray], is_center_crop=False) -> np.ndarray:
        if isinstance(image, str):
            image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if is_center_crop:
            # image = self.center_crop(image)
            image = self.center_crop_wa(image)
        image = cv2.resize(image, self.image_size).astype(np.float32)
        image = np.expand_dims(image, 0)
        return image


    def postprocess_prediction(self, raw_prediction: float) -> int:
        return 1 if (raw_prediction > self.threshold) else 0

    def center_crop(self, image: np.ndarray) -> np.ndarray:
        transform = A.Compose([
            A.CenterCrop(height=600, width=600, always_apply=True)
        ])
        return transform(image=image)['image']



class AntiSpoofingCosine(BaseONNXModel):

    def __init__(self, model_path: str, image_size: Tuple[int, int]):
        super().__init__(model_path=model_path, image_size=image_size)


    def load_single_image(self, image: Union[str, np.ndarray], is_center_crop=False) -> np.ndarray:
        if isinstance(image, str):
            image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if is_center_crop:
            # image = self.center_crop(image)
            image = self.center_crop_wa(image)
        image = cv2.resize(image, self.image_size).astype(np.float32)
        image = np.transpose(image, [2,0,1])
        image = image[None, ...]
        return image


    def postprocess_prediction(self, raw_prediction: float) -> int:
        return np.argmax(raw_prediction)

    def postprocess_prediction_as_text(self, raw_prediction: float) -> Tuple[int, str]:
        return np.argmax(raw_prediction), "Fake" if self.postprocess_prediction(raw_prediction) in [0,2] else "Real"