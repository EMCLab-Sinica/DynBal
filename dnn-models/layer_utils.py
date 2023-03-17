from __future__ import annotations

import logging
import math
import os
from typing import Any

import onnx

from utils import (
    find_initializer,
    find_tensor_value_info,
)
from configs import (
    ARM_PSTATE_LEN,
    OUTPUT_LEN,
    vm_size,
)

logger = logging.getLogger('intermittent-cnn.layer_utils')

def extend_for_footprints(batch_size, n):
    return n + n // batch_size

def determine_conv_tile_c(onnx_model: onnx.ModelProto, config: dict[str, Any], is_japari, intermediate_values_size, target, node, conv_flags):
    logger.debug('Determine tile size for Conv node %s', node.name)

    output_value_info = find_tensor_value_info(onnx_model, node.output[0])
    filter_info = find_initializer(onnx_model, node.input[1])

    shape = output_value_info.type.tensor_type.shape
    OUTPUT_CHANNEL = shape.dim[1].dim_value
    OUTPUT_H = shape.dim[2].dim_value
    OUTPUT_W = shape.dim[3].dim_value
    CHANNEL = filter_info.dims[1]
    kH = filter_info.dims[2]
    kW = filter_info.dims[3]

    max_continuous_channels = CHANNEL
    conv_flags.input_tile_c = max_continuous_channels

    logger.debug('Initial input_tile_c=%d', conv_flags.input_tile_c)

    def get_tile_input_usage(output_tile_c, filter_len):
        real_output_tile_c = output_tile_c
        # *2 as in JAPARI, the number of footprint weights is up to the number of
        # filters (e.g., batch size=1)
        if is_japari:
            real_output_tile_c *= 2
        ret = ((real_output_tile_c + 1) + 1) * filter_len
        return ret

    def get_pstate_usage(output_tile_c, filter_len):
        if target != 'msp432':
            return 0

        n_filters = output_tile_c
        if is_japari:
            n_filters *= 2
        return filter_len * n_filters

    while True:
        input_tile_too_large = False
        # inner +1 for biases
        filter_len = ((conv_flags.input_tile_c * kW + 1) + 1) // 2 * 2 * kH
        output_tile_c = OUTPUT_CHANNEL
        while True:
            tile_input_usage = get_tile_input_usage(output_tile_c, filter_len)
            pState_usage = get_pstate_usage(output_tile_c, filter_len)
            total_vm_usage = tile_input_usage + pState_usage
            logger.debug('Checking output_tile_c=%d, filter_len=%d, tile_input_usage=%d, pState_usage=%d, total_vm_usage=%d',
                         output_tile_c, filter_len, tile_input_usage, pState_usage, total_vm_usage)
            if ARM_PSTATE_LEN is not None and target == 'msp432':
                if tile_input_usage <= vm_size[target] - OUTPUT_LEN - ARM_PSTATE_LEN and pState_usage <= ARM_PSTATE_LEN:
                    break
            else:
                if total_vm_usage <= vm_size[target] - OUTPUT_LEN:
                    break
            logger.debug('output_tile_c=%d', output_tile_c)
            output_tile_c //= 2
            if output_tile_c % 2 or output_tile_c < config['op_filters']:
                # current input_tile_c is too large such that no even output_tile_c fits
                input_tile_too_large = True
                logger.debug("Input too large!")
                break

        if not input_tile_too_large:
            params_len = math.ceil(CHANNEL / conv_flags.input_tile_c) * OUTPUT_CHANNEL * OUTPUT_H * OUTPUT_W * 2
            if params_len <= intermediate_values_size:
                break
            logger.debug(f'params_len={params_len}, too high!')
        assert conv_flags.input_tile_c // 2 * 2 == conv_flags.input_tile_c
        conv_flags.input_tile_c //= 2
        logger.debug('input_tile_c=%d', conv_flags.input_tile_c)
    conv_flags.output_tile_c = output_tile_c

    reduce_output_ratio = float(os.getenv('TILE_SIZE_RATIO') or 1)
    conv_flags.output_tile_c = round(conv_flags.output_tile_c * reduce_output_ratio)
    conv_flags.output_tile_c = max(2, conv_flags.output_tile_c // 2 * 2)
    conv_flags.pState_len = pState_usage

    return output_tile_c

def get_gemm_pState_usage(tile_channel, tile_width, target):
    if target == 'msp432':
        return (tile_channel + 2) * (tile_width * 2)
    return 0

def check_gemm_vm_usage(A, tile_channel, tile_width, batch_size, target):
    A_shape = A.type.tensor_type.shape
    A_rows = 1  # Not using A_shape.dim[0] here, as it's a symbol "N"
    A_cols = A_shape.dim[1].dim_value

    full_tile_width = (extend_for_footprints(batch_size, tile_width)+1)/2*2
    tile_input_usage = (A_rows * A_cols + 2) + (tile_channel + 2) * full_tile_width + A_rows * full_tile_width
    pState_usage = get_gemm_pState_usage(tile_channel, tile_width, target)
    total_vm_usage = tile_input_usage + pState_usage

    ret = False
    if ARM_PSTATE_LEN is not None and target == 'msp432':
        if tile_input_usage <= vm_size[target] - ARM_PSTATE_LEN and pState_usage <= ARM_PSTATE_LEN:
            ret = True
    else:
        if total_vm_usage <= vm_size[target]:
            ret = True
    logger.debug("tile_channel=%d, tile_width=%d, tile_input_usage=%d, pState_usage=%d, total_vm_usage=%d => %s",
                 tile_channel, tile_width, tile_input_usage, pState_usage, total_vm_usage, "OK" if ret else "not OK")

    return ret

def determine_gemm_tile_sizes(onnx_model: onnx.ModelProto, config: dict[str, Any], batch_size, target, node, gemm_flags):
    logger.debug('Determine tile size for Gemm node %s', node.name)

    A = find_tensor_value_info(onnx_model, node.input[0])
    B = find_initializer(onnx_model, node.input[1])
    B_rows = B.dims[0]
    B_cols = B.dims[1]

    # writing a batch at a time is simpler and faster
    tile_size_unit = config['op_filters']

    gemm_flags.tile_width = tile_size_unit

    # LEA wants addresses to be 4 byte-aligned, or 2 Q15-aligned
    gemm_flags.tile_channel = min([B_rows,
                                   (config['gemm_tile_length'] or float('inf')),
                                   # MSP432 DMA controller only allows 1024 transfers for a DMA command. For external FRAM,
                                   # 1024 transfers = 1024 bytes = 512 Q-15 values
                                   512]) // tile_size_unit * tile_size_unit
    while True:
        if check_gemm_vm_usage(A, gemm_flags.tile_channel, gemm_flags.tile_width, batch_size, target):
            break
        assert gemm_flags.tile_channel > gemm_flags.tile_width
        gemm_flags.tile_channel -= gemm_flags.tile_width

    assert gemm_flags.tile_width % tile_size_unit == 0

    while True:
        new_tile_width = gemm_flags.tile_width + tile_size_unit
        if new_tile_width > B_cols:
            break
        if not check_gemm_vm_usage(A, gemm_flags.tile_channel, new_tile_width, batch_size, target):
            break
        gemm_flags.tile_width = new_tile_width

    gemm_flags.pState_len = get_gemm_pState_usage(gemm_flags.tile_channel, gemm_flags.tile_width, target)

    return gemm_flags.tile_width
