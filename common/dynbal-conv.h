#pragma once

#include <cstdint>
#include "dynbal.h"
#include "conv.h"

struct NodeFlags;
struct Node;

class UsageSpanConv : public UsageSpan {
public:
    UsageSpanConv(const ConvLayerDimensions& _layer_dims, uint16_t _input_tile_c, uint16_t _output_tile_c, uint32_t _power_cycle_energy)
        : layer_dims(_layer_dims)
        , input_tile_c(_input_tile_c)
        , output_tile_c(_output_tile_c)
        , input_tile_c_largest_local_minimum(_input_tile_c)
        , output_tile_c_largest_local_minimum(_output_tile_c)
        , power_cycle_energy(_power_cycle_energy)
    {
        output_tile_c_largest_local_minimum = nearest_value(ParameterDimension::OutputTileChannel, output_tile_c, /*not_larger_than=*/true);
        input_tile_c_largest_local_minimum = nearest_value(ParameterDimension::InputTileChannel, input_tile_c, /*not_larger_than=*/true);

        n_input_values = layer_dims.H * layer_dims.W * layer_dims.CHANNEL;
        n_filter_values = layer_dims.N_FILTERS * layer_dims.kH * layer_dims.kW * layer_dims.CHANNEL;
    }
    uint16_t nearest_value(uint8_t dim_idx, uint16_t dim_value, bool not_larger_than) const override;
    uint32_t data_reuse_cost(uint8_t dim_idx, uint16_t dim_value) const override;
    uint32_t data_refetch_cost(uint8_t dim_idx, uint16_t dim_value) const override;

    enum ParameterDimension {
        InputTileChannel,
        OutputTileChannel,
    };

private:
    const ConvLayerDimensions& layer_dims;
    uint16_t input_tile_c;
    uint16_t output_tile_c;
    uint16_t input_tile_c_largest_local_minimum;
    uint16_t output_tile_c_largest_local_minimum;
    uint32_t power_cycle_energy;

    uint32_t n_input_values;
    uint32_t n_filter_values;
};

void update_progress_indicator_conv(const Node* node, NodeFlags* flags, const NodeFlags* orig_flags, const ConvLayerDimensions& layer_dims, uint32_t first_unfinished_job_idx);
