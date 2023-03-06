import pathlib

import onnx
import onnx.shape_inference
import tensorflow as tf
import tf2onnx.convert

from model_utils import remove_tensorflow_input_transpose

def main():
    models_dir = pathlib.Path(__file__).resolve().parent

    # Load model and weights
    squeezenet_model_dir = models_dir / 'SqueezeNet_vs_CIFAR10' / 'models'
    with open(squeezenet_model_dir / 'squeeze_net.json') as f:
        model_json = f.read()
    model = tf.keras.models.model_from_json(model_json)
    model.load_weights(squeezenet_model_dir / 'squeeze_net.h5')

    onnx_model, _ = tf2onnx.convert.from_keras(model, opset=11)

    onnx_model = remove_tensorflow_input_transpose(onnx_model)

    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

    onnx.save(onnx_model, models_dir / 'squeezenet_cifar10.onnx')

if __name__ == '__main__':
    main()
