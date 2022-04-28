import pathlib

import onnx
import onnxoptimizer
import tensorflow as tf
import tf2onnx.convert

from utils import find_node_by_output

def main():
    models_dir = pathlib.Path(__file__).resolve().parent

    # Load model and weights
    squeezenet_model_dir = models_dir / 'SqueezeNet_vs_CIFAR10' / 'models'
    with open(squeezenet_model_dir / 'squeeze_net.json') as f:
        model_json = f.read()
    model = tf.keras.models.model_from_json(model_json)
    model.load_weights(squeezenet_model_dir / 'squeeze_net.h5')

    onnx_model, _ = tf2onnx.convert.from_keras(model)

    # Make input NCHW
    graph = onnx_model.graph
    first_conv = find_node_by_output(nodes=graph.node, output_name='squeezenet/conv/BiasAdd:0')
    first_conv.input[0] = 'input_1'
    input_dims = graph.input[0].type.tensor_type.shape.dim
    # Swap C and W
    input_dims[3].dim_value, input_dims[1].dim_value = input_dims[1].dim_value, input_dims[3].dim_value

    # Remove the now unused Transpose node
    onnx_model = onnxoptimizer.optimize(onnx_model, ['eliminate_deadend'])

    onnx.save(onnx_model, models_dir / 'squeezenet_cifar10.onnx')

if __name__ == '__main__':
    main()
