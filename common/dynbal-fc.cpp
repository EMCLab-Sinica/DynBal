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

// tile_channel: convex
// tile_width: monotonic

uint32_t UsageSpanFc::calc(uint8_t dim_idx, uint16_t dim_value) const {
    uint32_t n_input_values, n_filter_values;
    uint32_t input_fetch, filter_fetch, partial_sum_cost, data_reuse_cost;
    uint8_t n_tiles_c;

    uint16_t cur_tile_channel = (dim_idx == ParameterDimension::TileChannel) ? dim_value : tile_channel;
    uint16_t cur_tile_width = (dim_idx == ParameterDimension::TileWidth) ? dim_value : tile_width;
    n_input_values = layer_dims.A_rows * layer_dims.A_cols;
    n_filter_values = layer_dims.A_cols * layer_dims.B_cols;
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

    // Data refetch cost
    // TODO: compare with counters
    uint32_t input_cost, filter_cost, data_refetch_cost, usage_span;
    input_cost = cur_tile_channel * cur_tile_width;
    filter_cost = cur_tile_channel;
    // memory costs?
    data_refetch_cost = (input_cost * n_input_values + filter_cost * n_filter_values) / power_cycle_energy;
    usage_span = data_reuse_cost + data_refetch_cost;

    my_printf_debug("data_reuse_cost=%" PRIu32 " data_refetch_cost=%" PRIu32 " usage_span=%" PRIu32 NEWLINE, data_reuse_cost, data_refetch_cost, usage_span);

    return usage_span;
}

uint16_t UsageSpanFc::nearest_value(uint8_t dim_idx, uint16_t dim_value, bool not_larger_than) const {
    MY_ASSERT(dim_idx == ParameterDimension::TileChannel); // TODO: support TileWidth

    my_printf_debug("Finding the nearest local minimum for %d...", dim_value);
    uint16_t tmp;
    if (not_larger_than) {
        tmp = upper_gauss(layer_dims.A_cols, dim_value);
    } else {
        tmp = layer_dims.A_cols / dim_value;
    }
    // tile_channel should be multiple of op_filters, see determine_gemm_tile_sizes()
    uint16_t ret = (layer_dims.A_cols / tmp) / OP_FILTERS * OP_FILTERS;
    ret = LIMIT_DMA_SIZE(MIN_VAL(ret, tile_channel_largest_local_minimum));
    my_printf_debug("%d" NEWLINE, ret);
    return ret;
}

static void adapt_fc_dynbal(NodeFlags* node_flags, const NodeFlags* orig_flags, const UsageSpanFc* usage_span, const FcLayerDimensions& layer_dims, uint32_t jobs_in_a_power_cycle) {
    uint32_t output_len = layer_dims.A_rows * layer_dims.B_cols;
    uint16_t tile_channel_upper = orig_flags->gemm.tile_channel,
             tile_channel_lower = MAX_VAL(2, output_len * sizeof(int16_t) * layer_dims.A_cols / INTERMEDIATE_VALUES_SIZE);
    uint16_t tile_width_upper = orig_flags->gemm.tile_width, tile_width_lower = 2;
    const uint16_t value_ranges[2][2] = {
        { tile_channel_lower, tile_channel_upper },
        { tile_width_lower, tile_width_upper },
    };
    uint8_t dim_idx = UsageSpanFc::ParameterDimension::TileChannel;
    uint16_t new_tile_channel = convex_search(usage_span, dim_idx, value_ranges);
    node_flags->gemm.tile_channel = new_tile_channel;

    my_printf_debug("Selected tile_channel: %d" NEWLINE, node_flags->gemm.tile_channel);
}

void update_progress_indicator_fc(NodeFlags* node_flags, const NodeFlags* orig_flags, const FcLayerDimensions& layer_dims, uint32_t first_unfinished_value_offset) {
    InferenceStats* stats = load_inference_stats_from_nvm(InferenceStatsOpType::FC);

    const UsageSpanFc usage_span(layer_dims, orig_flags->gemm.tile_channel, orig_flags->gemm.tile_width, stats->power_cycle_energy);

    if (first_unfinished_value_offset == 0) {
        // Starting a new layer
        if (stats->power_cycle_energy) {
            adapt_fc_dynbal(node_flags, orig_flags, &usage_span, layer_dims, stats->power_cycle_energy);
            commit_node_flags(node_flags);
        } else {
            my_printf_debug("Skipping runtime reconfiguration!" NEWLINE);
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
