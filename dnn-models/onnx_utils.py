import onnx
import onnx.numpy_helper
import numpy as np

from utils import (
    find_initializer,
)

def add_tensor_annotation(onnx_model, key, tensor_name, data_type, vals):
    mapping = onnx.StringStringEntryProto()
    mapping.key = key
    mapping.value = f'{tensor_name}.{key}'

    annotation = onnx.TensorAnnotation()
    annotation.tensor_name = tensor_name
    annotation.quant_parameter_tensor_names.append(mapping)

    onnx_model.graph.quantization_annotation.append(annotation)

    vals = np.array(vals)
    tensor = onnx.helper.make_tensor(name=mapping.value, data_type=data_type,
                                     dims=np.shape(vals), vals=vals.flatten())
    onnx_model.graph.initializer.append(tensor)

def find_tensor_annotation(onnx_model: onnx.ModelProto, key: str, tensor_name: str):
    for tensor_annotation in onnx_model.graph.quantization_annotation:
        if tensor_annotation.tensor_name != tensor_name:
            continue
        for mapping in tensor_annotation.quant_parameter_tensor_names:
            if key != mapping.key:
                continue
            return onnx.numpy_helper.to_array(find_initializer(onnx_model, mapping.value))

def list_tensors_for_annotations(onnx_model: onnx.ModelProto):
    referenced_tensors = []
    for tensor_annotation in onnx_model.graph.quantization_annotation:
        for mapping in tensor_annotation.quant_parameter_tensor_names:
            referenced_tensors.append(mapping.value)
    return referenced_tensors

def get_param_limit(model: onnx.ModelProto, node: onnx.NodeProto):
    param_limit = 1
    for input_idx, input_ in enumerate(node.input[1:]):  # weights & possibly biases
        param_limit = max(param_limit, np.max(np.abs(onnx.numpy_helper.to_array(find_initializer(model, input_)))))
    return param_limit

def compute_parameter_scales(onnx_model: onnx.ModelProto):
    for node in onnx_model.graph.node:
        if node.op_type not in ('Conv', 'Gemm'):
            continue
        add_tensor_annotation(onnx_model, key='Q15_SCLAE_TENSOR', tensor_name=node.output[0],
                              data_type=onnx.TensorProto.DataType.FLOAT, vals=get_param_limit(onnx_model, node))
