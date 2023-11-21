import os

from antispoofing import AntiSpoofing, AntiSpoofingCosine

DATA_DIR = 'data'

if __name__ == '__main__':
    model_224 = AntiSpoofing('models/model_224.onnx', 0.9249477386474609, (224, 224))
    print('Start model with small tensors testing ...')
    for i in os.listdir(DATA_DIR):
        tens = model_224.load_single_image(os.path.join(DATA_DIR, i))
        prediction = model_224.run_model(tens)
        print(f'{i}, {prediction}, {model_224.postprocess_prediction(prediction)}')
    print('Done')

    print('Start model with big tensors testing ...')
    model_900 = AntiSpoofing('models/model_900.onnx', 0.5, (900, 900))
    for i in os.listdir(DATA_DIR):
        tens = model_900.load_single_image(os.path.join(DATA_DIR, i))
        prediction = model_900.run_model(tens)
        print(f'{i}, {prediction}, {model_900.postprocess_prediction(prediction)}')
    print('Done')

    print('Start model with cosine tensors testing ...')
    model_cosine = AntiSpoofingCosine('models/model_cosine_224.onnx', (224, 224))
    for i in os.listdir(DATA_DIR):
        tens = model_cosine.load_single_image(os.path.join(DATA_DIR, i))
        prediction = model_cosine.run_model(tens)
        print(f'{i}, {prediction}, {model_cosine.postprocess_prediction(prediction)}')
    print('Done')
