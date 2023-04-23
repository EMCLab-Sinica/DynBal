import dataclasses
import logging

import numpy

from utils import to_bytes
from model_utils import (
    dims_from_value_info,
    find_tensor_value_info,
)
from onnx_utils import (
    find_initializer,
)

logger = logging.getLogger('intermittent-cnn.dynbal')

NVM_RELATIVE_WRITE_COST = 1

@dataclasses.dataclass
class ConvLayerDimensions:
    H: int
    W: int
    CHANNEL: int
    OUTPUT_CHANNEL: int
    N_FILTERS: int
    kH: int
    kW: int
    OUTPUT_H: int
    OUTPUT_W: int

@dataclasses.dataclass
class FcLayerDimensions:
    A_rows: int
    A_cols: int
    B_cols: int

def usage_span_conv(layer_dims, cur_input_tile_c, cur_output_tile_c, power_cycle_energy, use_floor=True):
    NVM_READ_COST = int(numpy.floor((6*8*2)/(13/4)))

    upper_gauss = lambda x, y: (numpy.ceil(x / y) if use_floor else x / y)

    n_input_values = layer_dims.H * layer_dims.W * layer_dims.CHANNEL;
    n_filter_values = layer_dims.N_FILTERS * layer_dims.kH * layer_dims.kW * layer_dims.CHANNEL;
    n_one_filter_values = cur_input_tile_c * layer_dims.kH * layer_dims.kW;
    n_tiles_c = upper_gauss(layer_dims.CHANNEL, cur_input_tile_c);

    # Data reuse cost
    # OK: Verified by comparing fetch_cost with counter results
    n_filter_tiles_c = upper_gauss(layer_dims.N_FILTERS, cur_output_tile_c);
    input_fetch = layer_dims.kW * layer_dims.CHANNEL * layer_dims.H * layer_dims.OUTPUT_W * n_filter_tiles_c;
    filter_fetch = n_filter_values;
    partial_sum_cost = layer_dims.OUTPUT_H * layer_dims.OUTPUT_W * layer_dims.OUTPUT_CHANNEL * (
        (NVM_RELATIVE_WRITE_COST + 1) * n_tiles_c +         # writing and reading partial sums
        1                                                   # write complete output
    );
    data_reuse_cost = input_fetch + filter_fetch + partial_sum_cost;

    # Data refetch cost
    # TODO: compare with counters
    input_cost = cur_output_tile_c * n_one_filter_values;
    filter_cost = cur_output_tile_c * n_one_filter_values * layer_dims.OUTPUT_H * layer_dims.OUTPUT_W;
    # MEMORY_COST:
    input_cost += NVM_READ_COST * n_one_filter_values;
    filter_cost += NVM_READ_COST * n_one_filter_values * cur_output_tile_c;
    data_refetch_cost = (input_cost * n_input_values + filter_cost * n_filter_values) / power_cycle_energy;
    usage_span = data_reuse_cost + data_refetch_cost;

    #print("input_cost=%d filter_cost=%d partial_sum_cost=%d" % (input_cost, filter_cost, partial_sum_cost))
    #print("n_input_values=%d n_filter_values=%d" % (n_input_values, n_filter_values));
    #print("data_reuse_cost=%d data_refetch_cost=%d usage_span=%d" % (data_reuse_cost, data_refetch_cost, usage_span))

    return usage_span;

def usage_span_fc(layer_dims, cur_tile_channel, cur_tile_width, power_cycle_energy, use_floor=True):
    upper_gauss = lambda x, y: (numpy.ceil(x / y) if use_floor else x / y)

    n_input_values = layer_dims.A_rows * layer_dims.A_cols;
    n_filter_values = layer_dims.A_cols * layer_dims.B_cols;
    n_tiles_c = upper_gauss(layer_dims.A_cols, cur_tile_channel);

    # Data reuse cost
    # TODO: compare with counters
    input_fetch = layer_dims.A_rows * layer_dims.A_cols;
    filter_fetch = layer_dims.A_cols * layer_dims.B_cols;
    partial_sum_cost = layer_dims.A_rows * layer_dims.B_cols * (
        (NVM_RELATIVE_WRITE_COST + 1) * n_tiles_c +         # writing and reading partial sums
        1                                                   # write complete output
    );
    data_reuse_cost = input_fetch + filter_fetch + partial_sum_cost;

    # Data refetch cost
    # TODO: compare with counters
    input_cost = cur_tile_channel * cur_tile_width;
    filter_cost = cur_tile_channel;
    # memory costs?
    data_refetch_cost = (input_cost * n_input_values + filter_cost * n_filter_values) / power_cycle_energy;
    usage_span = data_reuse_cost + data_refetch_cost;

    return usage_span;

def parameter_importance_conv(onnx_model, node, power_cycle_energy):
    input_value_info = find_tensor_value_info(onnx_model, node.input[0])
    input_dims = dims_from_value_info(input_value_info)
    weights = find_initializer(onnx_model, node.input[1])
    output_value_info = find_tensor_value_info(onnx_model, node.output[0])
    output_dims = dims_from_value_info(output_value_info)

    # input and output dimension indices are decreased by 1 as the first dimension is unknown
    layer_dims = ConvLayerDimensions(
        H=input_dims[2-1],
        W=input_dims[3-1],
        CHANNEL=input_dims[1-1],
        OUTPUT_CHANNEL=output_dims[1-1],
        N_FILTERS=weights.dims[0],
        kH=weights.dims[2],
        kW=weights.dims[3],
        OUTPUT_H=output_dims[2-1],
        OUTPUT_W=output_dims[3-1],
    )

    cur_input_tile_c = input_dims[1-1]
    cur_output_tile_c = output_dims[1-1]
    cur_usage_span = usage_span_conv(layer_dims, cur_input_tile_c, cur_output_tile_c, power_cycle_energy)

    ind = numpy.arange(start=2, stop=cur_input_tile_c+2, step=2)
    usage_spans = usage_span_conv(layer_dims, ind, cur_output_tile_c, power_cycle_energy)
    input_tile_c_range = cur_usage_span - min(usage_spans)

    ind = numpy.arange(start=2, stop=cur_output_tile_c+2, step=2)
    usage_spans = usage_span_conv(layer_dims, cur_input_tile_c, ind, power_cycle_energy)
    output_tile_c_range = cur_usage_span - min(usage_spans)

    logger.debug('node %s: input_tile_c_range=%f, output_tile_c_range=%f',
                 node.name, input_tile_c_range, output_tile_c_range)
    if input_tile_c_range > output_tile_c_range:
        node.parameters_by_importance = [0, 1]
    else:
        node.parameters_by_importance = [1, 0]

def parameter_importance_fc(onnx_model, node, power_cycle_energy):
    input_value_info = find_tensor_value_info(onnx_model, node.input[0])
    input_dims = dims_from_value_info(input_value_info)
    weights = find_initializer(onnx_model, node.input[1])

    # input dimension indices are decreased by 1 as the first dimension is unknown
    layer_dims = FcLayerDimensions(
        A_rows=1,
        A_cols=input_dims[1-1],
        B_cols=weights.dims[1],
    )

    cur_tile_channel = input_dims[1-1]
    cur_tile_width = weights.dims[1]
    cur_usage_span = usage_span_fc(layer_dims, cur_tile_channel, cur_tile_width, power_cycle_energy)

    ind = numpy.arange(start=2, stop=cur_tile_channel+2, step=2)
    usage_spans = usage_span_fc(layer_dims, ind, cur_tile_width, power_cycle_energy)
    tile_channel_range = cur_usage_span - min(usage_spans)

    ind = numpy.arange(start=2, stop=cur_tile_width+2, step=2)
    usage_spans = usage_span_fc(layer_dims, cur_tile_channel, ind, power_cycle_energy)
    tile_width_range = cur_usage_span - min(usage_spans)

    logger.debug('node %s: tile_channel_range=%f, tile_width_range=%f',
                 node.name, tile_channel_range, tile_width_range)
    if tile_channel_range > tile_width_range:
        node.parameters_by_importance = [0, 1]
    else:
        node.parameters_by_importance = [1, 0]

def parameter_importance(onnx_model, nodes):
    power_cycle_energy = 100000

    for node in nodes:
        if node.op_type == 'Conv':
            parameter_importance_conv(onnx_model, node, power_cycle_energy)
        elif node.op_type == 'Gemm':
            parameter_importance_fc(onnx_model, node, power_cycle_energy)

def walk_search_space(nodes, node_flags, exhaustive_lookup_table):
    search_space_size = 0
    exhaustive_search_step = 4
    search_step = 2
    for node_idx, (node, cur_node_flags) in enumerate(zip(nodes, node_flags)):
        if node.op_type == 'Conv':
            for input_tile_c in range(exhaustive_search_step, cur_node_flags.conv.input_tile_c, exhaustive_search_step):
                exhaustive_lookup_table.write(to_bytes(node_idx))
                exhaustive_lookup_table.write(to_bytes(0))  # dim_idx
                exhaustive_lookup_table.write(to_bytes(input_tile_c))
            for output_tile_c in range(exhaustive_search_step, cur_node_flags.conv.output_tile_c, exhaustive_search_step):
                exhaustive_lookup_table.write(to_bytes(node_idx))
                exhaustive_lookup_table.write(to_bytes(1))  # dim_idx
                exhaustive_lookup_table.write(to_bytes(output_tile_c))
            search_space_size += int(cur_node_flags.conv.input_tile_c / search_step) * int(cur_node_flags.conv.output_tile_c / search_step)
        elif node.op_type == 'Gemm':
            for tile_channel in range(exhaustive_search_step, cur_node_flags.gemm.tile_channel, exhaustive_search_step):
                exhaustive_lookup_table.write(to_bytes(node_idx))
                exhaustive_lookup_table.write(to_bytes(0))  # dim_idx
                exhaustive_lookup_table.write(to_bytes(tile_channel))
            for tile_width in range(exhaustive_search_step, cur_node_flags.gemm.tile_width, exhaustive_search_step):
                exhaustive_lookup_table.write(to_bytes(node_idx))
                exhaustive_lookup_table.write(to_bytes(0))  # dim_idx
                exhaustive_lookup_table.write(to_bytes(tile_width))
            search_space_size += int(cur_node_flags.gemm.tile_channel / search_step) * int(cur_node_flags.gemm.tile_width / search_step)
    logger.info('Search space size: %d', search_space_size)
