from __future__ import annotations

import logging
import math
from typing import Any

import onnx

from utils import (
    find_initializer,
    find_tensor_value_info,
)
from configs import (
    ARM_PSTATE_LEN,
    lea_buffer_size,
)

logger = logging.getLogger('intermittent-cnn.layer_utils')

def extend_for_footprints(batch_size, n):
    return n + n // batch_size

def determine_conv_tile_c(onnx_model: onnx.ModelProto, config: dict[str, Any], is_japari, target, node):
    logger.debug('Determine tile size for Conv node %s', node.name)

    output_value_info = find_tensor_value_info(onnx_model, node.output[0])
    filter_info = find_initializer(onnx_model, node.input[1])
    node_flags = node.flags.conv

    shape = output_value_info.type.tensor_type.shape
    OUTPUT_CHANNEL = shape.dim[1].dim_value
    OUTPUT_H = shape.dim[2].dim_value
    OUTPUT_W = shape.dim[3].dim_value
    CHANNEL = filter_info.dims[1]
    kH = filter_info.dims[2]
    kW = filter_info.dims[3]

    max_continuous_channels = CHANNEL
    node_flags.input_tile_c = max_continuous_channels

    logger.debug('Initial input_tile_c=%d', node_flags.input_tile_c)

    def get_memory_usage(output_tile_c, filter_len):
        real_output_tile_c = output_tile_c
        # *2 as in JAPARI, the number of footprint weights is up to the number of
        # filters (e.g., batch size=1)
        if is_japari:
            real_output_tile_c *= 2
        ret = ((real_output_tile_c + 1) + 1) * filter_len
        logger.debug('Checking output_tile_c=%d, filter_len=%d, memory usage=%d', output_tile_c, filter_len, ret)
        return ret

    while True:
        input_tile_too_large = False
        # inner +1 for biases
        filter_len = ((node_flags.input_tile_c * kW + 1) + 1) // 2 * 2 * 2 * kH
        output_tile_c = OUTPUT_CHANNEL
        while get_memory_usage(output_tile_c, filter_len) > lea_buffer_size[target]:
            logger.debug('output_tile_c=%d', output_tile_c)
            output_tile_c //= 2
            if output_tile_c % 2 or output_tile_c < config['op_filters']:
                # current input_tile_c is too large such that no even output_tile_c fits
                input_tile_too_large = True
                logger.debug("Input too large!")
                break

        if not input_tile_too_large:
            params_len = math.ceil(CHANNEL / node_flags.input_tile_c) * OUTPUT_CHANNEL * OUTPUT_H * OUTPUT_W * 2
            if params_len < config['intermediate_values_size']:
                break
            logger.debug(f'params_len={params_len}, too high!')
        assert node_flags.input_tile_c / 2 * 2 == node_flags.input_tile_c
        node_flags.input_tile_c //= 2
        logger.debug('input_tile_c=%d', node_flags.input_tile_c)
    node_flags.output_tile_c = output_tile_c
    return output_tile_c

def determine_gemm_tile_sizes(onnx_model: onnx.ModelProto, config: dict[str, Any], batch_size, target, node):
    logger.debug('Determine tile size for Gemm node %s', node.name)

    A = find_tensor_value_info(onnx_model, node.input[0])
    B = find_initializer(onnx_model, node.input[1])
    A_shape = A.type.tensor_type.shape
    A_rows = 1  # Not using A_shape.dim[0] here, as it's a symbol "N"
    A_cols = A_shape.dim[1].dim_value
    B_rows = B.dims[0]
    node_flags = node.flags.gemm

    # writing a batch at a time is simpler and faster
    tile_size_unit = config['op_filters']

    while True:
        # LEA wants addresses to be 4 byte-aligned, or 2 Q15-aligned
        node_flags.tile_channel = min([(ARM_PSTATE_LEN // (tile_size_unit * 2)) // 2 * 2 - 2, B_rows,
                                       (config['gemm_tile_length'] or float('inf')),
                                       # MSP432 DMA controller only allows 1024 transfers for a DMA command. For external FRAM,
                                       # 1024 transfers = 1024 bytes = 512 Q-15 values
                                       512]) // tile_size_unit * tile_size_unit
        full_tile_width = (extend_for_footprints(batch_size, tile_size_unit)+1)/2*2
        while node_flags.tile_channel > 0:
            tmp = int(math.ceil(B_rows / node_flags.tile_channel))
            needed_mem = (A_rows * A_cols + 2) + (node_flags.tile_channel + 2) * full_tile_width + A_rows * full_tile_width
            logger.debug("tile_channel=%d, tmp=%d, needed_mem=%d", node_flags.tile_channel, tmp, needed_mem)
            if needed_mem <= lea_buffer_size[target]:
                break
            node_flags.tile_channel -= tile_size_unit
        logger.debug("tile_channel = %d", node_flags.tile_channel)
        if node_flags.tile_channel > 0:
            break

    assert (tile_size_unit * 2) * (node_flags.tile_channel + 2) <= ARM_PSTATE_LEN
    return tile_size_unit
