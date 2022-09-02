from __future__ import annotations

import enum
import itertools
import logging
import os.path
import pathlib
import sys
import tarfile
import zipfile
from typing import Callable, Iterable, NamedTuple, Optional
from urllib.request import urlretrieve

import filelock
import numpy as np
import onnx
import onnxoptimizer
import onnxruntime
import onnxruntime.backend as backend
import platformdirs
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger('intermittent-cnn.utils')

INPLACE_UPDATE_OPS = ['Reshape', 'Softmax', 'Squeeze', 'Unsqueeze']
OPS_WITH_MERGE = ['Conv', 'Gemm']

THIS_DIR = pathlib.Path(__file__).absolute().parent

audio_ops = ['DecodeWav', 'AudioSpectrogram', 'Mfcc']

class DataLayout(enum.Enum):
    NEUTRAL = 0
    NCW = 1
    NWC = 2
    NCHW = 3
    NHWC = 4

class ModelData(NamedTuple):
    dataset: Dataset
    data_layout: DataLayout

    def data_loader(self, limit):
        return DataLoader(self.dataset, batch_size=(limit or len(self.dataset)))

def extract_archive(archive_path: pathlib.Path, subdir: str):
    archive_dir = archive_path.with_name(subdir)
    if not archive_dir.exists():
        if '.tar' in str(archive_path):
            with tarfile.open(archive_path) as tar:
                members = [member for member in tar.getmembers() if member.name.startswith(subdir)]
                tar.extractall(archive_path.parent, members=members)
        elif str(archive_path).endswith('.zip'):
            with zipfile.ZipFile(archive_path) as zip_f:
                members = [member for member in zip_f.namelist() if member.startswith(subdir)]
                zip_f.extractall(archive_path.parent, members=members)
    return archive_dir

def kws_dnn_model():
    return download_file('https://github.com/ARM-software/ML-KWS-for-MCU/raw/master/Pretrained_models/DNN/DNN_S.pb', 'KWS-DNN_S.pb')

def download_file(url: str, filename: str, post_processor: Optional[Callable] = None) -> os.PathLike:
    xdg_cache_home = platformdirs.user_cache_path()

    lock_path = xdg_cache_home / f'{filename}.lock'

    # Inspired by https://stackoverflow.com/a/53643011
    class ProgressHandler:
        def __init__(self):
            self.last_reported = 0

        def __call__(self, block_num, block_size, total_size):
            progress = int(block_num * block_size / total_size * 100)
            if progress > self.last_reported + 5:
                logger.info('Downloaded: %d%%', progress)
                self.last_reported = progress

    with filelock.FileLock(lock_path):
        local_path = xdg_cache_home / filename
        if not local_path.exists():
            urlretrieve(url, local_path, ProgressHandler())

        ret = local_path
        if post_processor:
            ret = post_processor(local_path)

    return ret

def find_initializer(onnx_model: onnx.ModelProto, name: str) -> Optional[onnx.TensorProto]:
    for initializer in onnx_model.graph.initializer:
        if initializer.name == name:
            return initializer

def find_tensor_value_info(onnx_model: onnx.ModelProto, name: str) -> onnx.ValueInfoProto:
    if name.endswith('_before_merge'):
        name = name[:-len('_before_merge')]
    g = onnx_model.graph
    for value_info in itertools.chain(g.value_info, g.input, g.output):
        if value_info.name == name:
            return value_info
    raise ValueError(f'No value_info found for {name}')

def find_node_by_output(nodes: list[onnx.NodeProto], output_name: str) -> onnx.NodeProto:
    for node in nodes:
        for output in node.output:
            if output == output_name:
                return node

def find_node_by_input(nodes: list[onnx.NodeProto], input_name: str) -> onnx.NodeProto:
    for node in nodes:
        for input_ in node.input:
            if input_ == input_name:
                return node

def get_attr(node, attr_name):
    for attr in node.attribute:
        if attr.name != attr_name:
            continue
        return onnx.helper.get_attribute_value(attr)

    # Not found
    return None

def find_kernel_shape(model, node):
    kernel_shape = get_attr(node, 'kernel_shape')
    if not kernel_shape:
        if node.op_type == 'MaxPool': # this field is required for maxpool
            raise Exception('kernel_shape is required for MaxPool')
        weights = node.input[1]
        w = find_initializer(model, weights)
        kernel_shape = w.dims[2:]
    assert len(kernel_shape) == 2
    return kernel_shape

def infer_auto_pad(model, node):
    # https://github.com/onnx/onnx/blob/master/docs/Operators.md#conv
    auto_pad = get_attr(node, 'auto_pad')
    pads = get_attr(node ,'pads') or [0]*4
    assert len(pads) <= 4
    if auto_pad in (b'SAME_UPPER', b'SAME_LOWER'):
        kernel_shape = find_kernel_shape(model, node)
        pads[0] = pads[2] = kernel_shape[0] // 2
        pads[1] = pads[3] = kernel_shape[1] // 2
        if pads[0]*2+1 != kernel_shape[0] or pads[1]*2+1 != kernel_shape[1]:
            raise NotImplementedError
    return pads

def numpy_type_to_onnx_elem_type(numpy_type):
    if numpy_type == np.float32:
        return onnx.TensorProto.FLOAT
    if numpy_type == np.int64:
        return onnx.TensorProto.INT64
    if numpy_type == np.bool_:
        return onnx.TensorProto.BOOL
    raise Exception(f'Unsupported type {numpy_type}')

def get_model_ops(onnx_model):
    # Retrieving information for operators. Inspired by the script for generating
    # https://github.com/onnx/onnx/blob/v1.10.2/docs/Operators.md [1,2]
    # [1] https://github.com/onnx/onnx/blob/v1.10.2/onnx/defs/gen_doc.py
    # [2] https://github.com/onnx/onnx/blob/v1.10.2/onnx/onnx_cpp2py_export/defs.pyi
    ops = set()
    for schema in onnx.defs.get_all_schemas():
        ops.add(schema.name)

    ops = ops.intersection(node.op_type for node in onnx_model.graph.node)
    for op in OPS_WITH_MERGE:
        if op in ops:
            ops.add(op + 'Merge')
    ops = sorted(ops)

    return ops

def load_model(config, model_variant):
    model_name = config['onnx_model']
    if model_variant:
        model_name += f'-{model_variant}'
    # https://github.com/onnx/onnx/blob/master/docs/PythonAPIOverview.md
    onnx_model = onnx.load_model(THIS_DIR / f'{model_name}.onnx')

    # onnxoptimizer requires known dimensions, so set the batch size=1.
    # The batch size will be changed to a variable after dynamic_shape_inference, anyway.
    # https://github.com/onnx/optimizer/blob/v0.2.6/onnxoptimizer/passes/fuse_matmul_add_bias_into_gemm.h#L60
    change_batch_size(onnx_model)

    # https://zhuanlan.zhihu.com/p/41255090
    onnx_model = onnxoptimizer.optimize(onnx_model, [
        'eliminate_nop_dropout',
        'extract_constant_to_initializer',
        'fuse_add_bias_into_conv',
        'fuse_matmul_add_bias_into_gemm',
    ])

    dynamic_shape_inference(onnx_model, config['sample_size'])
    onnx.checker.check_model(onnx_model)

    return onnx_model

def add_merge_nodes(model):
    # Split Conv/Gemm into Conv/Gemm and ConvMerge/GemmMerge (for merging OFMs from channel tiling)
    new_nodes = []
    for idx, n in enumerate(model.graph.node):
        if n.op_type in audio_ops:
            logger.warning('skipping audio operator %s', n.op_type)
            continue
        new_nodes.append(n)
        if n.op_type in OPS_WITH_MERGE:
            output_name = n.output[0]
            new_node = onnx.NodeProto()
            new_node.name = (n.name or n.op_type) + ':merge'
            new_node.op_type = n.op_type + 'Merge'
            new_node.input[:] = n.output[:] = [output_name + '_before_merge']
            new_node.output[:] = [output_name]
            new_nodes.append(new_node)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

def onnxruntime_prepare_model(model):
    return backend.prepare(onnxruntime.InferenceSession(
        model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    ))

def onnxruntime_get_intermediate_tensor(model, image):
    # Creating a new model with all nodes as outputs
    # https://github.com/microsoft/onnxruntime/issues/1455#issuecomment-979901463
    tmp_model = onnx.ModelProto()
    tmp_model.CopyFrom(model)

    orig_outputs = list(tmp_model.graph.output)
    orig_output_names = [node.name for node in orig_outputs]
    del tmp_model.graph.output[:]
    for node in tmp_model.graph.node:
        for output in node.output:
            if output not in orig_output_names:
                tmp_model.graph.output.append(onnx.ValueInfoProto(name=output))
    tmp_model.graph.output.extend(orig_outputs)

    rep = onnxruntime_prepare_model(tmp_model)
    outputs = rep.run(image)
    for idx, output in enumerate(outputs):
        output_name = tmp_model.graph.output[idx].name
        node = find_node_by_output(tmp_model.graph.node, output_name)
        yield output_name, node.op_type, output

def change_batch_size(onnx_model: onnx.ModelProto):
    g = onnx_model.graph
    initializer_names = set([initializer.name for initializer in g.initializer])
    constant_names = set([node.output[0] for node in g.node if node.op_type == 'Constant'])
    for value_info in itertools.chain(g.value_info, g.input, g.output):
        if value_info.name in initializer_names or value_info.name in constant_names:
            continue
        shape = value_info.type.tensor_type.shape
        if shape.dim and shape.dim[0].dim_param:
            shape.dim[0].dim_value = 1

    # make sure above steps did not break the model
    onnx.shape_inference.infer_shapes(onnx_model)

def dynamic_shape_inference(onnx_model: onnx.ModelProto, sample_size: Iterable[int]) -> None:
    for node in itertools.chain(onnx_model.graph.input, onnx_model.graph.output):
        if not node.type.tensor_type.shape.dim:
            continue
        node.type.tensor_type.shape.dim[0].dim_param = 'N'

    del onnx_model.graph.value_info[:]

    BATCH_SIZE = 2  # Any number larger than 1 is OK. Here I pick the smallest one for performance considerations

    dummy_images = np.expand_dims(np.zeros(sample_size, dtype=np.float32), axis=0)
    shapes = {
        layer_name: np.shape(layer_out)
        for layer_name, _, layer_out in onnxruntime_get_intermediate_tensor(onnx_model, dummy_images)
    }
    dummy_images = np.concatenate([
        np.expand_dims(np.random.rand(*sample_size).astype(np.float32), axis=0) for _ in range(BATCH_SIZE)
    ], axis=0)

    value_infos = []
    for layer_name, layer_type, layer_out in onnxruntime_get_intermediate_tensor(onnx_model, dummy_images):
        larger_shape = np.shape(layer_out)
        smaller_shape = shapes[layer_name]
        if larger_shape[1:] != smaller_shape[1:]:
            logger.info('Skipping OFM %s for %s node with mismatched shapes: %r, %r', layer_name, layer_type, larger_shape, smaller_shape)
            continue

        new_shape = list(larger_shape)
        if larger_shape:
            if larger_shape[0] == smaller_shape[0] * BATCH_SIZE:
                new_shape[0] = 'N'
            elif larger_shape[0] == smaller_shape[0]:
                pass
            else:
                logger.info('Skipping OFM %s for %s node with mismatched batch sizes: %d, %d', layer_name, layer_type, larger_shape[0], smaller_shape[0])
                continue

        elem_type = numpy_type_to_onnx_elem_type(layer_out.dtype)
        value_info = onnx.helper.make_tensor_value_info(layer_name, elem_type, new_shape)
        value_infos.append(value_info)

    onnx_model.graph.value_info.extend(value_infos)

def print_float(val):
    print('%13.6f' % val, end='')

def print_tensor(tensor, print_histogram):
    shape = np.shape(tensor)
    print(f'Shape: {shape}')
    dimensions = np.shape(shape)[0]
    if dimensions == 4:
        N, C, H, W = shape
        assert N == 1
        for c in range(C):
            print(f'Channel {c}')
            for h in range(H):
                for w in range(W):
                    print_float(tensor[0, c, h, w])
                print()
            print()
    elif dimensions == 3:
        N, C, W = shape
        for n in range(N):
            for c in range(C):
                print(f'Channel {c}')
                for w in range(W):
                    print_float(tensor[n, c, w])
                print()
                print()
            print()
    elif dimensions == 2:
        H, W = shape
        for h in range(H):
            for w in range(W):
                print_float(tensor[h, w])
            print()
    elif dimensions == 1:
        if shape[0] >= 1024:
            print(f'Skipping very long vector with length {shape[0]}')
            return
        for idx in range(shape[0]):
            print_float(tensor[idx])
            if idx % 16 == 15:
                print()
        print()
    else:
        print(f'Skip: unsupported {dimensions}-dimensional array')
    if dimensions >= 1 and np.prod(shape) != 0:
        if print_histogram:
            threshold = 1
            abs_tensor = np.absolute(tensor)
            total = np.prod(tensor.shape)
            while True:
                count = np.count_nonzero(np.where(abs_tensor >= threshold, tensor, np.zeros(tensor.shape)))
                if not count:
                    break
                print(f'>= {threshold}: {count} / {100.0*count/total:.2f}%')
                threshold *= 2
        print(f'Max={np.max(tensor)}, min={np.min(tensor)}')

def run_model(model, model_data, limit, verbose=True, save_file=None):
    # Testing
    images, labels = next(iter(model_data.data_loader(limit)))
    images = images.numpy()
    if limit == 1:
        last_layer_out = None
        if verbose:
            print('Input')
            print_tensor(images, False)
        if save_file:
            model_output_pb2 = import_model_output_pb2()
            model_output = model_output_pb2.ModelOutput()
        for layer_name, op_type, layer_out in onnxruntime_get_intermediate_tensor(model, images):
            if verbose:
                print(f'{op_type} layer: {layer_name}')
                print_tensor(layer_out, op_type in ('Conv', 'Gemm'))
            if save_file:
                layer_out_obj = model_output_pb2.LayerOutput()
                layer_out_obj.name = layer_name
                layer_out_obj.dims.extend(layer_out.shape)
                if layer_out.shape:
                    linear_shape = [np.prod(layer_out.shape)]
                    layer_out_obj.value.extend(np.reshape(layer_out, linear_shape))
                else:
                    # zero-dimension tensor -> scalar
                    layer_out_obj.value.append(layer_out)
                model_output.layer_out.append(layer_out_obj)
            # Softmax is not implemented yet - return the layer before Softmax
            if op_type != 'Softmax':
                last_layer_out = layer_out
        if save_file:
            with open(save_file, 'wb') as f:
                f.write(model_output.SerializeToString())
        return last_layer_out
    else:
        correct = 0
        layer_outs = onnxruntime_prepare_model(model).run(images)[0]
        for idx, layer_out in enumerate(layer_outs):
            predicted = np.argmax(layer_out)
            if predicted == labels[idx]:
                if verbose:
                    print(f'Correct at idx={idx}')
                correct += 1
        total = len(labels)
        accuracy = correct/total
        if verbose:
            print(f'correct={correct} total={total} rate={accuracy}')
        return accuracy

def remap_inputs(model: onnx.ModelProto, input_mapping: dict[str, str]):
    new_inputs = list(input_mapping.values())
    for new_input in new_inputs:
        model.graph.input.append(onnx.ValueInfoProto(name=new_input))
    for node in model.graph.node:
        node.input[:] = [input_mapping.get(inp, inp) for inp in node.input]
        node.output[:] = [
            output + '_unused' if output in new_inputs else output
            for output in node.output
        ]
    for idx, inp in enumerate(model.graph.input):
        if inp.name in input_mapping.keys():
            del model.graph.input[idx]

    return onnxoptimizer.optimize(model, ['eliminate_deadend'])

def import_model_output_pb2():
    try:
        orig_sys_path = sys.path.copy()
        sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / 'build'))
        import model_output_pb2
        return model_output_pb2
    finally:
        sys.path = orig_sys_path
