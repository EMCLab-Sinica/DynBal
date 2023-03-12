import argparse
import dataclasses
import io
import itertools
import logging
import os.path
import pathlib
import textwrap
import warnings

import cffi
import onnx
import onnx.defs
import onnx.helper
import onnx.numpy_helper
import numpy as np

from configs import (
    configs,
    vm_size,
    OUTPUT_LEN,
)
from utils import (
    DataLayout,
    INPLACE_UPDATE_OPS,
    THIS_DIR,
    add_merge_nodes,
    get_attr,
    find_kernel_shape,
    find_initializer,
    find_node_by_input,
    find_node_and_idx_by_output,
    get_model_ops,
    infer_auto_pad,
    load_model,
    run_model,
    to_bytes,
)
from onnx_utils import (
    compute_parameter_scales,
    find_tensor_annotation,
    get_sample_size,
    list_tensors_for_annotations,
)
from model_utils import (
    find_min_range,
)
from layer_utils import (
    determine_conv_tile_c,
    determine_gemm_tile_sizes,
)

logging.basicConfig()
logger = logging.getLogger('intermittent-cnn.transform')

"""
Goal: Mapping name-based nodes to integer-based ones.
Indexing policy:
    0~len(onnx_model.graph.input)-1: input nodes
    len(onnx_model.graph.input)~ : other (hidden) nodes
"""

class Constants:
    SLOT_PARAMETERS = 0xfe
    SLOT_TEST_SET = 0xff
    NODE_NAME_LEN = 60
    TURNING_POINTS_LEN = 8
    MODEL_NODES_LEN = 0
    INPUTS_DATA_LEN = 0
    NUM_INPUTS = 0  # will be filled during parsing
    N_INPUT = 0
    # Match the size of external FRAM
    NVM_SIZE = 512 * 1024
    INTERMEDIATE_VALUES_SIZE = 0  # will be filled by nvm_layout()
    N_SAMPLES = 20
    LEA_BUFFER_SIZE = 0
    OUTPUT_LEN = OUTPUT_LEN
    USE_ARM_CMSIS = 0
    CONFIG = None

    BATCH_SIZE = 1
    STATEFUL = 0
    HAWAII = 0
    JAPARI = 0
    INTERMITTENT = 0
    INDIRECT_RECOVERY = 0
    METHOD = "Baseline"
    FIRST_SAMPLE_OUTPUTS = []
    USE_STATES_ARRAY = 0

other_flags = [
    # parameter flags
    'CHANNEL_FIRST',
    'TRANSPOSED',
]

def op_flag(flag):
    return 2 ** other_flags.index(flag)

def _Q15(arr, name):
    """Transform a floating point number to TI's fixed point _q15 format"""

    # See DSPLib_1_30_00_02/include/DSPLib_support.h

    lower = -1
    upper = 32767.0 / 32768.0

    overflowed_indices = np.concatenate((
        np.flatnonzero(np.asarray(arr < lower)),
        np.flatnonzero(np.asarray(arr > upper))
    ))
    for idx in overflowed_indices:
        warnings.warn(f'{name} value {arr[idx]} goes beyond the range of _q15 ({lower}, {upper})')

    arr = np.minimum(np.maximum(arr, lower), upper)

    return (arr * 2 ** 15).astype(int)

def init_cffi():
    ffi = cffi.FFI()

    c_sources = ''
    with open(THIS_DIR.parent / 'common' / 'data_structures.h') as f:
        for line in f:
            if line.startswith(('#include', 'static_assert')):
                continue
            c_sources += line
    ffi.cdef(c_sources)
    return ffi

ffi = init_cffi()

class ONNXNodeWrapper:
    def __init__(self, orig_node: onnx.NodeProto):
        self.orig_node = orig_node
        self.max_output_id = 0
        self.name = orig_node.name or orig_node.output[0] or orig_node.op_type
        self.inputs = []

    def __getattr__(self, name):
        return getattr(self.orig_node, name)


def get_prev_node(n):
    return nodes[names[n.input[0]] - Constants.N_INPUT]

parser = argparse.ArgumentParser()
parser.add_argument('config', choices=configs.keys())
parser.add_argument('--all-samples', action='store_true')
parser.add_argument('--write-images', action='store_true')
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--target', choices=('msp430', 'msp432'), required=True)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--data-output-dir', metavar='DIR', default='build')
parser.add_argument('--model-variant', type=str, default='')
intermittent_methodology = parser.add_mutually_exclusive_group(required=True)
intermittent_methodology.add_argument('--ideal', action='store_true')
intermittent_methodology.add_argument('--hawaii', action='store_true')
intermittent_methodology.add_argument('--japari', action='store_true')
intermittent_methodology.add_argument('--stateful', action='store_true')
args = parser.parse_args()
if args.debug:
    logging.getLogger('intermittent-cnn').setLevel(logging.DEBUG)
else:
    logging.getLogger('intermittent-cnn').setLevel(logging.INFO)

config = configs[args.config]
onnx_model = load_model(config, model_variant=args.model_variant)

sample_size = get_sample_size(onnx_model)

config['total_sample_size'] = np.prod(sample_size)
if 'gemm_tile_length' not in config:
    config['gemm_tile_length'] = 0
Constants.CONFIG = args.config
if args.all_samples:
    Constants.N_SAMPLES = config['n_all_samples']
    Constants.NVM_SIZE += config['n_all_samples'] * 2*config['total_sample_size']  # multiply by 2 for Q15
model_data = config['data_loader'](train=False, target_size=sample_size)
images, labels = next(iter(model_data.data_loader(limit=Constants.N_SAMPLES)))
images = images.numpy()

Constants.FIRST_SAMPLE_OUTPUTS = list(run_model(onnx_model, model_data, limit=1, verbose=False)[0])
Constants.FP32_ACCURACY = run_model(onnx_model, model_data, limit=None, verbose=False)
add_merge_nodes(onnx_model)

Constants.BATCH_SIZE = args.batch_size
if args.stateful:
    Constants.STATEFUL = 1
    Constants.METHOD = "STATEFUL"
if args.hawaii:
    Constants.HAWAII = 1
    Constants.METHOD = "HAWAII"
if args.japari:
    Constants.JAPARI = 1
    Constants.METHOD = "JAPARI"
Constants.INTERMITTENT = Constants.STATEFUL | Constants.HAWAII | Constants.JAPARI
Constants.INDIRECT_RECOVERY = Constants.STATEFUL | Constants.JAPARI
if args.target == 'msp432':
    Constants.USE_ARM_CMSIS = 1
Constants.LEA_BUFFER_SIZE = vm_size[args.target]

names = {}

# Remove Squeeze and Reshape nodes with constants as the input
replaced_nodes_map = {}

def replace_squeeze(node, inp):
    # Since opset 13, axes is an input instead of an attribute
    try:
        axes_name = node.input[1]
        axes = find_initializer(onnx_model, axes_name).int64_data
    except IndexError:
        axes = get_attr(node, 'axes')
    new_dims = [dim for dim_idx, dim in enumerate(inp.dims) if dim_idx not in axes]
    # Repeated fields cannot be assigned directly
    # https://developers.google.com/protocol-buffers/docs/reference/python-generated#repeated-fields
    inp.dims[:] = new_dims

def replace_reshape(node, inp):
    dims_name = node.input[1]
    new_dims = find_initializer(onnx_model, dims_name).int64_data
    assert new_dims
    inp.dims[:] = new_dims

replace_handlers = {
    'Squeeze': replace_squeeze,
    'Reshape': replace_reshape,
}

def replace_nodes():
    for n in onnx_model.graph.node:
        if n.op_type not in ('Squeeze', 'Reshape'):
            continue
        inp = find_initializer(onnx_model, n.input[0])
        if inp:
            replace_handlers[n.op_type](n, inp)
            replaced_nodes_map[n.output[0]] = n.input[0]

def transpose_gemm(onnx_model: onnx.ModelProto):
    for node in onnx_model.graph.node:
        if node.op_type != 'Gemm':
            continue
        transB = get_attr(node, 'transB')
        B = find_initializer(onnx_model, node.input[1])
        if transB != 1 or B is None:
            continue
        data = onnx.numpy_helper.to_array(B)
        data = np.transpose(data)
        B.CopyFrom(onnx.helper.make_tensor(B.name, B.data_type, (B.dims[1], B.dims[0]), np.concatenate(data)))
        for idx, attr in enumerate(node.attribute):
            if attr.name == 'transB':
                del node.attribute[idx]
                break

replace_nodes()
transpose_gemm(onnx_model)

new_nodes = [n for n in onnx_model.graph.node if n.output[0] not in replaced_nodes_map.keys()]
for n in new_nodes:
    for idx, inp in enumerate(n.input):
        n.input[idx] = replaced_nodes_map.get(inp, inp)

nodes = [ONNXNodeWrapper(n) for n in new_nodes]
node_flags = ffi.new('struct NodeFlags[]', len(nodes))
for cur_node_flags in node_flags:
    cur_node_flags.canary = 0x55

conv_param_names = set()

for idx, inp in enumerate(onnx_model.graph.input):
    names[inp.name] = idx

# For some ONNX models (e.g., squeezenet-cifar10 converted from Keras), inputs
# do not include initializers. Merge them here.
inputs_len = len(names.keys())
for idx, initializer in enumerate(onnx_model.graph.initializer):
    if initializer.name not in names:
        names[initializer.name] = idx + inputs_len

compute_parameter_scales(onnx_model)

Constants.N_INPUT = len(names.keys())
logger.info('Constants.N_INPUT = %d', Constants.N_INPUT)

for idx, n in enumerate(nodes):
    if n.op_type == 'Dropout':
        output = n.output[:1]  # we don't care the second output `mask`
    else:
        output = n.output
    if n.op_type == 'Conv':
        conv_param_names.add(n.input[1])
        node_flags[idx].conv.pads = infer_auto_pad(onnx_model, n)
        node_flags[idx].conv.group = get_attr(n, 'group') or 1
    if n.op_type in ('Conv', 'MaxPool'):
        extra_flags = getattr(node_flags[idx], n.op_type.lower())
        kernel_shape = find_kernel_shape(onnx_model, n)
        extra_flags.kernel_shape = kernel_shape
        strides = get_attr(n, 'strides')
        if strides is not None:
            extra_flags.strides = strides
        else:
            # "If not present, the stride defaults to 1 along each spatial axis."
            # https://github.com/onnx/onnx/blob/main/docs/Operators.md#Conv
            # https://github.com/onnx/onnx/blob/main/docs/Operators.md#maxpool
            extra_flags.strides = (1, 1)
    if n.op_type == 'MaxPool':
        ceil_mode = get_attr(n, 'ceil_mode')
        if ceil_mode:
            node_flags[idx].maxpool.ceil = 1
    if n.op_type == 'Reshape':
        prev_node = n
        while prev_node and prev_node.op_type in INPLACE_UPDATE_OPS:
            prev_node_idx, prev_node = find_node_and_idx_by_output(nodes, prev_node.input[0])
        if prev_node and prev_node.op_type == 'MaxPool':
            node_flags[prev_node_idx].maxpool.nhwc2nchw = 1
    if n.op_type in ('Squeeze', 'Unsqueeze'):
        axes = get_attr(n, 'axes') or []
        node_flags[idx].squeeze.axes = 0
        for axis in axes:
            node_flags[idx].squeeze.axes |= (1 << axis)
    if n.op_type == 'GemmMerge':
        node_flags[idx].gemmmerge.tile_length = config['gemm_tile_length']
    if n.op_type == 'Concat':
        node_flags[idx].concat.axis = get_attr(n, 'axis')
    for output_ in output:
        names[output_] = idx + Constants.N_INPUT

for idx, node in enumerate(nodes):
    node.inputs = [names[i] for i in node.input]
    for inp in node.inputs:
        if inp < Constants.N_INPUT:
            continue
        used_node = nodes[inp - Constants.N_INPUT]
        used_node.max_output_id = max([idx, used_node.max_output_id])

parameters = [None for _ in range(Constants.N_INPUT)]

tensors_referenced_in_annotations = list_tensors_for_annotations(onnx_model)
for params in onnx_model.graph.initializer:
    if params.data_type not in (onnx.TensorProto.FLOAT, onnx.TensorProto.INT64):
        raise Exception('unsupported data type {}'.format(params.data_type))
    if params.name in tensors_referenced_in_annotations:
        continue

    assert parameters[names[params.name]] is None
    parameters[names[params.name]] = params

def nchw2nhwc(arr, dims):
    arr = np.reshape(arr, dims)  # Change flattened to 4-D
    arr = np.transpose(arr, axes=(0, 2, 3, 1))  # NCHW -> NHWC
    return arr.flatten()  # Change it back to flattened

outputs = {
    'parameters': io.BytesIO(),
    'samples': io.BytesIO(),
    'model': io.BytesIO(),
    'nodes': io.BytesIO(),
    'node_flags': [],
    'node_orig_flags': [],
    'footprints': [],
    'inference_stats': [],
    'model_parameters_info': io.BytesIO(),
    'intermediate_parameters_info': io.BytesIO(),
    'labels': io.BytesIO(),
}

Constants.MODEL_NODES_LEN = len(nodes)

model = outputs['model']
model.write(to_bytes(0))  # Model.running
model.write(to_bytes(0))  # Model.run_counter
model.write(to_bytes(0))  # Model.layer_idx
for _ in range(config['num_slots']): # Model.slots_info
    if Constants.INDIRECT_RECOVERY:
        model.write(to_bytes(1, size=8)) # SlotInfo.state_bit
        model.write(to_bytes(0, size=8)) # SlotInfo.n_turning_points
        for __ in range(Constants.TURNING_POINTS_LEN):
            model.write(to_bytes(-1))   # SlotInfo.turning_points
    model.write(to_bytes(-1))       # SlotInfo.user
model.write(to_bytes(0, size=8))  # Model.dummy
model.write(to_bytes(0, size=8))  # Model.version

@dataclasses.dataclass
class ParametersSlot:
    offset: int
    target: io.BytesIO
    slot_id: int

parameters_slot = ParametersSlot(offset=0, target=outputs['parameters'], slot_id=Constants.SLOT_PARAMETERS)

output_nodes = outputs['nodes']
for node in nodes:
    Constants.NUM_INPUTS = max(Constants.NUM_INPUTS, len(node.inputs))
logger.info('Maximum number of inputs = %d', Constants.NUM_INPUTS)

ops = get_model_ops(onnx_model)

def write_str(buffer: io.BytesIO, data: str):
    assert Constants.NODE_NAME_LEN >= len(data), f'String too long: {data}'
    buffer.write(data.encode('ascii') + b'\0' * (Constants.NODE_NAME_LEN - len(data)))

for node in nodes:
    write_str(output_nodes, node.name)
    write_str(output_nodes, node.output[0])
    output_nodes.write(to_bytes(len(node.inputs)))
    for inp in node.inputs:
        output_nodes.write(to_bytes(inp))
    for _ in range(Constants.NUM_INPUTS - len(node.inputs)):
        output_nodes.write(to_bytes(0))
    output_nodes.write(to_bytes(node.max_output_id))
    output_nodes.write(to_bytes(ops.index(node.op_type)))

footprints_arr = ffi.new('struct Footprint[]', 2 * len(nodes))
outputs['footprints'].append(footprints_arr)

inference_stats_arr = ffi.new('struct InferenceStats[]', 2)
outputs['inference_stats'].append(inference_stats_arr)

outputs['node_orig_flags'].append(node_flags)

# Two copies for shadowing
for _ in range(2):
    outputs['node_flags'].append(node_flags)

parameter_info_idx = 0

def write_scale(dest, scale):
    shift = 0
    while scale >= 1:
        shift += 1
        scale /= 2
    dest.write(to_bytes(int(scale*2**15)))             # scale.fract
    dest.write(to_bytes(shift, size=8))     # scale.shift
    dest.write(to_bytes(0, size=8))         # scale.dummy

model_parameters_info = outputs['model_parameters_info']
total_params = 0
for params in parameters:
    if params is None:  # input
        # Actual data for test samples are added last
        dims = images[0].shape
        model_parameters_info.write(to_bytes(parameters_slot.offset, size=32))  # params_offset
        model_parameters_info.write(to_bytes(np.prod(dims) * 2, size=32))  # A _q15 is 16-bit
        model_parameters_info.write(to_bytes(Constants.SLOT_TEST_SET, size=8))     # slot
        model_parameters_info.write(to_bytes(0, size=8))     # param_flags
        # extend_dims
        model_parameters_info.write(to_bytes(1))
        for dim in dims:
            model_parameters_info.write(to_bytes(dim))
        for _ in range(3 - len(dims)):
            model_parameters_info.write(to_bytes(0))
        write_scale(model_parameters_info, config['input_scale'])
    else:
        assert len(params.dims) <= 4
        params_data = onnx.numpy_helper.to_array(params)
        model_parameters_info.write(to_bytes(parameters_slot.offset, size=32))  # params_offset
        if params.data_type == onnx.TensorProto.FLOAT:
            param_size = 2
            if params.name in conv_param_names:
                logger.info('Reorder conv param %s', params.name)
                params_data = nchw2nhwc(params_data, params.dims)
            used_node = find_node_by_input(onnx_model.graph.node, params.name)

            if used_node.op_type == 'Gemm':
                params_data = np.reshape(params_data, params.dims)
                params_data = np.transpose(params_data)

            param_scale = find_tensor_annotation(onnx_model, key='Q15_SCLAE_TENSOR', tensor_name=params.name) or config['scale']
            parameters_slot.target.write(to_bytes(_Q15(params_data / param_scale, 'Parameter')))
        elif params.data_type == onnx.TensorProto.INT64:
            param_size = 8
            for param in params_data:
                parameters_slot.target.write(to_bytes(param, size=64))
        else:
            assert False
        data_len = np.prod(params.dims)
        parameters_slot.offset += data_len * param_size
        model_parameters_info.write(to_bytes(data_len * param_size, size=32))
        model_parameters_info.write(to_bytes(parameters_slot.slot_id, size=8))  # slot
        model_parameters_info.write(to_bytes(0, size=8))  # param_flags
        if len(params.dims) == 4:
            channels = params.dims[1]
        else:
            channels = 0
        logger.info('dims = %r, length = %d', params.dims, data_len)
        for dim in params.dims:
            model_parameters_info.write(to_bytes(dim))
        # dims are always 4 uint16_t's in C++
        for _ in range(4 - len(params.dims)):
            model_parameters_info.write(to_bytes(0))
        write_scale(model_parameters_info, param_scale)
        total_params += data_len

    # common to input and non-inputs
    model_parameters_info.write(to_bytes(parameter_info_idx))        # parameter_info_idx
    parameter_info_idx += 1

logger.info('Total params: %d', total_params)

# Placeholder for ParameterInfo of intermediate values
intermediate_parameters_info = outputs['intermediate_parameters_info']
for idx, n in enumerate(nodes):
    intermediate_parameters_info.write(to_bytes(0, size=32))  # params_offset
    intermediate_parameters_info.write(to_bytes(0, size=32))  # params_len
    intermediate_parameters_info.write(to_bytes(0, size=8))  # slot
    intermediate_parameters_info.write(to_bytes(0, size=8))  # param_flags
    for _ in range(4):  # dims[4]
        intermediate_parameters_info.write(to_bytes(0))
    intermediate_parameters_info.write(to_bytes(0, size=32))   # scale
    intermediate_parameters_info.write(to_bytes(parameter_info_idx))             # parameter_info_idx
    parameter_info_idx += 1

def ensure_channel_last(images, data_layout):
    if data_layout in (DataLayout.NEUTRAL, DataLayout.NHWC, DataLayout.NWC):
        return images
    elif data_layout == DataLayout.NCW:
        return np.transpose(images, axes=(0, 2, 1))  # NCW => NWC
    elif data_layout == DataLayout.NCHW:
        return np.transpose(images, axes=(0, 2, 3, 1))  # NCHW => NHWC
    else:
        raise NotImplementedError

images = ensure_channel_last(images, model_data.data_layout)
for idx in range(images.shape[0]):
    im = images[idx, :]
    outputs['samples'].write(to_bytes(_Q15(im.flatten(order='C') / config['input_scale'], 'Input')))
    if args.write_images:
        import cv2
        os.makedirs('images', exist_ok=True)
        # Restore conanical image format (H, W, C)
        im = np.squeeze(im * 256)
        cv2.imwrite(f'images/test{idx:02d}.png', im)

for label in labels:
    outputs['labels'].write(to_bytes(label, size=8))

if args.write_images:
    with open('images/ans.txt', 'w') as f:
        f.write(' '.join(map(str, labels)))

def nvm_layout():
    # See common/platform.h; some items are duplicated for double buffering
    nvm_data_names = ['inference_stats', 'model', 'model', 'intermediate_parameters_info', 'node_flags', 'nodes', 'footprints', 'parameters']
    remaining_size = Constants.NVM_SIZE - 256
    for data_name in nvm_data_names:
        if isinstance(outputs[data_name], io.BytesIO):
            cur_data_size = outputs[data_name].tell()
        else:
            cur_data_size = sum(ffi.sizeof(item) for item in outputs[data_name])
        logger.debug('Data size for %s: %d', data_name, cur_data_size)
        remaining_size -= cur_data_size
    # Size for samples are different for plat-pc and plat-mcu
    remaining_size -= 2*config['total_sample_size']
    # intermediate_values_size should < 65536, or TI's compiler gets confused
    Constants.INTERMEDIATE_VALUES_SIZE = int((remaining_size / config['num_slots']) / 16) * 16
    if args.target == 'msp430':
        Constants.INTERMEDIATE_VALUES_SIZE = min(Constants.INTERMEDIATE_VALUES_SIZE, 65534)
    logger.debug('INTERMEDIATE_VALUES_SIZE=%d', Constants.INTERMEDIATE_VALUES_SIZE)

nvm_layout()

max_output_tile_size = 0
for idx, n in enumerate(nodes):
    if n.op_type == 'Conv':
        cur_output_tile_c = determine_conv_tile_c(onnx_model, config, Constants.JAPARI, Constants.INTERMEDIATE_VALUES_SIZE, args.target, n,  node_flags[idx].conv)
        max_output_tile_size = max(max_output_tile_size, cur_output_tile_c)
    if n.op_type == 'Gemm':
        cur_output_tile_c = determine_gemm_tile_sizes(onnx_model, config, Constants.BATCH_SIZE, args.target, n, node_flags[idx].gemm)
        max_output_tile_size = max(max_output_tile_size, cur_output_tile_c)

if Constants.STATEFUL:
    min_range = find_min_range(onnx_model, nodes, node_flags, config, Constants.N_INPUT)
    if min_range < max_output_tile_size:
        Constants.USE_STATES_ARRAY = 1

pathlib.Path(args.data_output_dir).mkdir(exist_ok=True)

with open(f'{args.data_output_dir}/data.cpp', 'w') as output_c, open(f'{args.data_output_dir}/data.h', 'w') as output_h:
    output_h.write('''
#pragma once

#include <stdint.h>

struct ParameterInfo;
struct Model;
struct Node;
struct NodeFlags;

''')
    for item in itertools.chain(dir(Constants), config.keys()):
        if hasattr(Constants, item):
            if item.startswith('__'):
                continue
            val = getattr(Constants, item)
        else:
            val = config[item]
            # Somehow for integers, numpy.array uses int64 on Linux and int32 on Windows
            if not isinstance(val, (int, float, np.int64, np.int32)):
                continue
        # Making it long to avoid overflow for expressions like
        # INTERMEDIATE_VALUES_SIZE * NUM_SLOTS on 16-bit systems
        suffix = 'l' if item == 'INTERMEDIATE_VALUES_SIZE' else ''
        output_h.write(f'#define {item.upper()} ')
        if isinstance(val, str):
            output_h.write(f'"{val}"')
        elif isinstance(val, list):
            output_h.write('{' + ', '.join(map(str, val)) + '}')
        else:
            output_h.write(f'{val}')
        output_h.write(f'{suffix}\n')

    output_c.write('''
#include "data.h"
#include "cnn_common.h"
#include "platform.h"
''')

    func_params = ','.join((
        'struct Model *model',
        'const struct ParameterInfo *input[]',
        'struct ParameterInfo *output',
        'const struct Node* node',
        'struct NodeFlags* node_flags',
        'const struct NodeFlags* orig_node_flags',
    ))
    # ops
    output_h.write('\n')
    for idx, op in enumerate(ops):
        output_h.write(f'#define Op{op} {idx}\n')

    for op in ops:
        output_h.write('void alloc_{}({});\n'.format(op.lower(), func_params))
        output_h.write('void handle_{}({});\n'.format(op.lower(), func_params))
    output_c.write('const handler handlers[] = {\n')
    for op in ops:
        output_c.write(f'    handle_{op},\n'.lower())
    output_c.write('};\n')
    output_c.write('const allocator allocators[] = {\n')
    for op in ops:
        output_c.write(f'    alloc_{op},\n'.lower())
    output_c.write('};\n')
    for op in ops:
        if op in INPLACE_UPDATE_OPS:
            output_c.write(textwrap.dedent(f'''
                void alloc_{op.lower()}({func_params}) {{
                    SlotInfo *cur_slot_info = get_slot_info(model, output->slot);
                    if (cur_slot_info) {{
                        cur_slot_info->user = model->layer_idx;
                    }}
                }}
            '''))
        else:
            output_c.write(textwrap.dedent(f'''
                #if defined(FALLBACK_HANDLERS) && (defined(__GNUC__) || defined(__clang__))
                void __attribute__((weak)) alloc_{op.lower()}({func_params}) {{
                    ERROR_OCCURRED();
                }}
                #endif
            '''))
        output_c.write(textwrap.dedent(f'''
            #if defined(FALLBACK_HANDLERS) && (defined(__GNUC__) || defined(__clang__))
            void __attribute__((weak)) handle_{op.lower()}({func_params}) {{
                ERROR_OCCURRED();
            }}
            #endif
        '''))

    # data
    for idx, name in enumerate(other_flags):
        output_h.write(f'#define {name} {2**idx}\n')

    def hex_str(arr):
        return '  ' + ', '.join([f'0x{num:02x}' for num in arr]) + ',\n'

    def define_var(var_name, data):
        output_h.write(f'''
extern const uint8_t * const {var_name};
#define {var_name.upper()}_LEN {len(data)}
''')
        # #define with _Pragma seems to be broken :/
        output_c.write(f'''
const uint8_t _{var_name}[{len(data)}] = {{
''')
        n_pieces, remaining = divmod(len(data), 16)
        for idx in range(n_pieces):
            output_c.write(hex_str(data[idx*16:(idx+1)*16]))
        if remaining:
            output_c.write(hex_str(data[len(data) - remaining:len(data)]))
        output_c.write(f'''}};
const uint8_t * const {var_name} = _{var_name};
''')

    for var_name, data_obj in outputs.items():
        full_var_name = var_name + '_data'
        if isinstance(data_obj, io.BytesIO):
            data_obj.seek(0)
            if full_var_name == 'samples_data':
                data = data_obj.read(2*config['total_sample_size'])
            else:
                data = data_obj.read()
        else:  # a list of ffi objects
            data = b''
            for item in data_obj:
                data += ffi.buffer(item)
        define_var(full_var_name, data)

with open('samples.bin', 'wb') as f:
    samples = outputs['samples']
    samples.seek(0)
    f.write(samples.read())
