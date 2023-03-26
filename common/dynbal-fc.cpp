#include <cinttypes>
#include <cstdint>

#include "cnn_common.h"
#include "data.h"
#include "data_structures.h"
#include "dynbal-fc.h"
#include "fc.h"
#include "layers.h"
#include "op_utils.h"
#include "my_debug.h"
#include "platform.h"

// tile_channel: convex
// tile_width: convex

uint32_t UsageSpanFc::data_reuse_cost(uint8_t dim_idx, uint16_t dim_value) const {
    uint32_t input_fetch, filter_fetch, partial_sum_cost, data_reuse_cost;
    uint8_t n_tiles_c;

    uint16_t cur_tile_channel = (dim_idx == ParameterDimension::TileChannel) ? dim_value : tile_channel;
    n_tiles_c = upper_gauss(layer_dims.A_cols, cur_tile_channel);

    my_printf_debug("cur_tile_channel=%d n_tiles_c=%d" NEWLINE, cur_tile_channel, n_tiles_c);

    // Data reuse cost
    // TODO: compare with counters
    input_fetch = layer_dims.A_rows * layer_dims.A_cols;
    filter_fetch = layer_dims.A_cols * layer_dims.B_cols;
    partial_sum_cost = layer_dims.A_rows * layer_dims.B_cols * (
        (NVM_RELATIVE_WRITE_COST + 1) * n_tiles_c +         // writing and reading partial sums
        1                                                   // write complete output
    );
    data_reuse_cost = input_fetch + filter_fetch + partial_sum_cost;

    my_printf_debug("data_reuse_cost=%" PRIu32 NEWLINE, data_reuse_cost);

    return data_reuse_cost;
}

uint32_t UsageSpanFc::data_refetch_cost(uint8_t dim_idx, uint16_t dim_value) const {
    uint16_t cur_tile_channel = (dim_idx == ParameterDimension::TileChannel) ? dim_value : tile_channel;
    uint16_t cur_tile_width = (dim_idx == ParameterDimension::TileWidth) ? dim_value : tile_width;

    // Data refetch cost
    // TODO: compare with counters
    uint32_t input_cost, filter_cost, data_refetch_cost;
    input_cost = cur_tile_channel * cur_tile_width;
    filter_cost = cur_tile_channel;
    // memory costs?
    data_refetch_cost = (input_cost * n_input_values + filter_cost * n_filter_values) / power_cycle_energy;

    my_printf_debug("data_refetch_cost=%" PRIu32 NEWLINE, data_refetch_cost);

    return data_refetch_cost;
}

uint16_t UsageSpanFc::nearest_value(uint8_t dim_idx, uint16_t dim_value, bool not_larger_than) const {
    my_printf_debug("Finding the nearest local minimum for %d...", dim_value);
    uint16_t tmp, dim_original_value, dim_upper_bound;
    if (dim_idx == ParameterDimension::TileChannel) {
        dim_original_value = layer_dims.A_cols;
        dim_upper_bound = tile_channel_largest_local_minimum;
    } else {
        dim_original_value = layer_dims.B_cols;
        dim_upper_bound = tile_width_largest_local_minimum;
    }
    if (not_larger_than) {
        tmp = upper_gauss(dim_original_value, dim_value);
    } else {
        tmp = dim_original_value / dim_value;
    }
    // tile_channel should be multiple of op_filters, see determine_gemm_tile_sizes()
    uint16_t ret = (dim_original_value / tmp) / OP_FILTERS * OP_FILTERS;
    ret = LIMIT_DMA_SIZE(MIN_VAL(ret, dim_upper_bound));
    my_printf_debug("ret=%d" NEWLINE, ret);
    return ret;
}

static void adapt_fc_dynbal(const Node* node, NodeFlags* node_flags, const NodeFlags* orig_flags, const UsageSpanFc* usage_span, const FcLayerDimensions& layer_dims, uint32_t jobs_in_a_power_cycle) {
    uint32_t output_len = layer_dims.A_rows * layer_dims.B_cols;
    uint16_t tile_channel_upper = orig_flags->gemm.tile_channel,
             tile_channel_lower = MAX_VAL(2, output_len * sizeof(int16_t) * layer_dims.A_cols / INTERMEDIATE_VALUES_SIZE);
    uint16_t tile_width_upper = orig_flags->gemm.tile_width, tile_width_lower = 2;
    const uint16_t value_ranges[2][2] = {
        { tile_channel_lower, tile_channel_upper },
        { tile_width_lower, tile_width_upper },
    };
    for (uint8_t dim_idx : node->parameters_by_importance) {
        uint16_t new_dim_value = convex_search(usage_span, dim_idx, value_ranges);
        if (!read_gpio_flag(GPIOFlag::DisableDynBalReconfiguration)) {
            if (dim_idx == UsageSpanFc::ParameterDimension::TileChannel) {
                node_flags->gemm.tile_channel = new_dim_value;
                my_printf_debug("Selected tile_channel: %d" NEWLINE, node_flags->gemm.tile_channel);
            } else {
                node_flags->gemm.tile_width = new_dim_value;
                my_printf_debug("Selected tile_width: %d" NEWLINE, node_flags->gemm.tile_width);
            }
        }
    }
}

void update_progress_indicator_fc(const Node* node, NodeFlags* node_flags, const NodeFlags* orig_flags, const FcLayerDimensions& layer_dims, uint32_t first_unfinished_value_offset) {
    if (read_gpio_flag(GPIOFlag::DisableDynBalTracking)) {
        return;
    }

    InferenceStats* stats = load_inference_stats_from_nvm(InferenceStatsOpType::FC);

    const UsageSpanFc usage_span(layer_dims, orig_flags->gemm.tile_channel, orig_flags->gemm.tile_width, stats->power_cycle_energy);

    if (first_unfinished_value_offset == 0) {
        if (!read_gpio_flag(GPIOFlag::DisableDynBalSearch)) {
            // Starting a new layer
            if (stats->power_cycle_energy) {
                adapt_fc_dynbal(node, node_flags, orig_flags, &usage_span, layer_dims, stats->power_cycle_energy);
                commit_node_flags(node_flags);
            } else {
                my_printf_debug("Skipping runtime reconfiguration!" NEWLINE);
            }
        }
        // Cleanup stats from the previous layer
        stats->last_progress_indicator = 0;
        commit_inference_stats(InferenceStatsOpType::FC);
    } else {
        uint32_t last_job_idx = stats->last_progress_indicator;
        stats->last_progress_indicator = first_unfinished_value_offset;

        uint32_t n_elapsed_jobs = first_unfinished_value_offset - last_job_idx;
        uint32_t macs = n_elapsed_jobs * node_flags->gemm.tile_channel;

        stats->power_cycle_energy = macs;
        my_printf_debug("macs=%" PRIu32, macs);
        // memory costs?
        my_printf_debug(NEWLINE);

        commit_inference_stats(InferenceStatsOpType::FC);
    }
}
