#include <cinttypes>
#include <cmath>
#include "cnn_common.h"
#include "conv.h"
#include "dynbal-conv.h"
#include "layers.h"
#include "my_debug.h"
#include "op_utils.h"
#include "platform.h"

// output_tile_c: convex
// input_tile_c: convex

static const uint32_t NVM_READ_COST = FRAM_FREQ_DIVIDER*8*sizeof(int16_t)*4/13;

uint32_t UsageSpanConv::data_reuse_cost(uint8_t dim_idx, uint16_t dim_value) const {
    uint32_t input_fetch, filter_fetch, partial_sum_cost, data_reuse_cost;
    uint8_t n_tiles_c, n_filter_tiles_c;

    uint16_t cur_input_tile_c = (dim_idx == ParameterDimension::InputTileChannel) ? dim_value : input_tile_c;
    uint16_t cur_output_tile_c = (dim_idx == ParameterDimension::OutputTileChannel) ? dim_value : output_tile_c;
    n_tiles_c = upper_gauss(layer_dims.CHANNEL, cur_input_tile_c);

    // Data reuse cost
    // OK: Verified by comparing fetch_cost with counter results
    n_filter_tiles_c = upper_gauss(layer_dims.N_FILTERS, cur_output_tile_c);
    input_fetch = layer_dims.kW * layer_dims.CHANNEL * layer_dims.H * layer_dims.OUTPUT_W * n_filter_tiles_c;
    filter_fetch = n_filter_values;
    partial_sum_cost = layer_dims.OUTPUT_H * layer_dims.OUTPUT_W * layer_dims.OUTPUT_CHANNEL * (
        (NVM_RELATIVE_WRITE_COST + 1) * n_tiles_c +         // writing and reading partial sums
        1                                                   // write complete output
    );
    data_reuse_cost = input_fetch + filter_fetch + partial_sum_cost;

    my_printf_debug("partial_sum_cost=%" PRIu32 NEWLINE, partial_sum_cost);
    my_printf_debug("data_reuse_cost=%" PRIu32 NEWLINE, data_reuse_cost);

    return data_reuse_cost;
}

uint32_t UsageSpanConv::data_refetch_cost(uint8_t dim_idx, uint16_t dim_value) const {
    uint32_t n_one_filter_values;

    uint16_t cur_input_tile_c = (dim_idx == ParameterDimension::InputTileChannel) ? dim_value : input_tile_c;
    uint16_t cur_output_tile_c = (dim_idx == ParameterDimension::OutputTileChannel) ? dim_value : output_tile_c;

    n_one_filter_values = cur_input_tile_c * layer_dims.kH * layer_dims.kW;

    // Data refetch cost
    // TODO: compare with counters
    uint64_t input_cost, filter_cost, data_refetch_cost;
    input_cost = cur_output_tile_c * n_one_filter_values;
    filter_cost = cur_output_tile_c * n_one_filter_values * layer_dims.OUTPUT_H * layer_dims.OUTPUT_W;
    // memory costs
    input_cost += NVM_READ_COST * n_one_filter_values;
    filter_cost += NVM_READ_COST * n_one_filter_values * cur_output_tile_c;
    data_refetch_cost = (input_cost * n_input_values + filter_cost * n_filter_values) / power_cycle_energy;

    my_printf_debug("input_cost=%" PRIu64 " filter_cost=%" PRIu64 NEWLINE, input_cost, filter_cost);
    my_printf_debug("data_refetch_cost=%" PRIu64 NEWLINE, data_refetch_cost);

    return data_refetch_cost;
}

uint16_t UsageSpanConv::nearest_value(uint8_t dim_idx, uint16_t dim_value, bool not_larger_than) const {
    my_printf_debug("Finding the nearest local minimum for %d...", dim_value);
    uint16_t tmp, dim_original_value, dim_upper_bound;
    if (dim_idx == ParameterDimension::OutputTileChannel) {
        dim_original_value = layer_dims.N_FILTERS;
        dim_upper_bound = output_tile_c_largest_local_minimum;
    } else {
        dim_original_value = layer_dims.CHANNEL;
        dim_upper_bound = input_tile_c_largest_local_minimum;
    }
    if (not_larger_than) {
        tmp = upper_gauss(dim_original_value, dim_value);
    } else {
        tmp = dim_original_value / dim_value;
    }
    uint16_t ret = MIN_VAL((dim_original_value / tmp) / 2 * 2, dim_upper_bound);
    my_printf_debug("ret=%d" NEWLINE, ret);
    return ret;
}

#if RuntimeConfiguration == DynBal
void adapt_conv_dynbal(const Node* node, NodeFlags* node_flags, const NodeFlags* orig_flags, const UsageSpanConv* usage_span, uint32_t power_cycle_energy) {
    uint16_t output_tile_c_upper = orig_flags->conv.output_tile_c, output_tile_c_lower = 2;
    uint16_t input_tile_c_upper = orig_flags->conv.input_tile_c, input_tile_c_lower = 2;
    const uint16_t value_ranges[2][2] = {
        { input_tile_c_lower,  input_tile_c_upper },
        { output_tile_c_lower, output_tile_c_upper }
    };
    for (uint8_t dim_idx : node->parameters_by_importance) {
        uint16_t new_dim_value = convex_search(usage_span, dim_idx, value_ranges);
        if (!read_gpio_flag(GPIOFlag::DisableDynBalReconfiguration)) {
            if (dim_idx == UsageSpanConv::ParameterDimension::OutputTileChannel) {
                node_flags->conv.output_tile_c = new_dim_value;
                my_printf_debug("Selected output_tile_c: %d" NEWLINE, new_dim_value);
            } else {
                node_flags->conv.input_tile_c = new_dim_value;
                my_printf_debug("Selected input_tile_c: %d" NEWLINE, new_dim_value);
            }
        }
    }
}

void update_progress_indicator_conv(const Node* node, NodeFlags* node_flags, const NodeFlags* orig_flags, const ConvLayerDimensions& layer_dims, uint32_t first_unfinished_job_idx) {
    if (read_gpio_flag(GPIOFlag::DisableDynBalTracking)) {
        return;
    }

    InferenceStats* stats = load_inference_stats_from_nvm(InferenceStatsOpType::Conv);

    const UsageSpanConv usage_span(layer_dims, orig_flags->conv.input_tile_c, orig_flags->conv.output_tile_c, stats->power_cycle_energy);

    if (first_unfinished_job_idx == 0) {
        if (!read_gpio_flag(GPIOFlag::DisableDynBalSearch)) {
            // Starting a new layer
            if (stats->power_cycle_energy) {
                adapt_conv_dynbal(node, node_flags, orig_flags, &usage_span, stats->power_cycle_energy);
                commit_node_flags(node_flags);
            } else {
                my_printf_debug("Skipping runtime reconfiguration!" NEWLINE);
            }
        }
        // Cleanup stats from the previous layer
        stats->last_progress_indicator = 0;
        commit_inference_stats(InferenceStatsOpType::Conv);
    } else {
        uint32_t last_job_idx = stats->last_progress_indicator;
        stats->last_progress_indicator = first_unfinished_job_idx;

        uint32_t n_elapsed_jobs = first_unfinished_job_idx - last_job_idx;
        uint32_t macs = n_elapsed_jobs * node_flags->conv.input_tile_c * layer_dims.kH * layer_dims.kW;

        // OK: verified with counters
        stats->power_cycle_energy = macs;
        my_printf_debug("macs=%" PRIu32, macs);
        // memory costs
        uint32_t output_strip_size = node_flags->conv.output_tile_c * layer_dims.OUTPUT_H * layer_dims.OUTPUT_W;
        uint32_t filter_tile_size = node_flags->conv.output_tile_c * layer_dims.kH * layer_dims.kW * node_flags->conv.input_tile_c;
        uint16_t n_filter_tiles = first_unfinished_job_idx / output_strip_size - last_job_idx / output_strip_size + 1; // # of filter tiles involved
                                                                                                                       //
        uint32_t filter_reads = filter_tile_size * n_filter_tiles;
        uint32_t input_reads = (n_elapsed_jobs / node_flags->conv.output_tile_c) *  // the number of input tiles
                               (node_flags->conv.input_tile_c * MIN_VAL(layer_dims.kH, layer_dims.STRIDE_H) * layer_dims.kW);
        stats->power_cycle_energy += (filter_reads + input_reads) * NVM_READ_COST;
        my_printf_debug(" filter_reads=%" PRIu32 " input_reads=%" PRIu32,
                        filter_reads, input_reads);
        my_printf_debug(NEWLINE);

        commit_inference_stats(InferenceStatsOpType::Conv);
    }
}
#endif

