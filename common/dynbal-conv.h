#pragma once

#include <cstdint>
#include "dynbal.h"
#include "conv.h"

class NodeFlags;

class UsageSpanConv : public UsageSpan {
public:
    UsageSpanConv(const ConvLayerDimensions& _layer_dims, uint16_t _input_tile_c, uint16_t _output_tile_c, uint32_t _power_cycle_energy)
        : layer_dims(_layer_dims)
        , input_tile_c(_input_tile_c)
        , output_tile_c(_output_tile_c)
        , power_cycle_energy(_power_cycle_energy)
    {}
    uint16_t nearest_value(uint8_t dim_idx, uint16_t dim_value) const;
    uint32_t calc(uint8_t dim_idx, uint16_t dim_value) const;

    enum ParameterDimension {
        InputTileChannel,
        OutputTileChannel,
    };
protected:
    uint32_t data_reuse_cost() const;

private:
    const ConvLayerDimensions& layer_dims;
    uint16_t input_tile_c;
    uint16_t output_tile_c;
    uint32_t power_cycle_energy;
};

void update_progress_indicator_conv(NodeFlags* flags, const NodeFlags* orig_flags, const ConvLayerDimensions& layer_dims, uint32_t first_unfinished_job_idx);