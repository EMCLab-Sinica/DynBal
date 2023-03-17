#pragma once

#include <cstdint>
#include "dynbal.h"
#include "fc.h"

struct NodeFlags;
struct Node;

class UsageSpanFc : public UsageSpan {
public:
    UsageSpanFc(const FcLayerDimensions& _layer_dims, uint16_t _tile_channel, uint16_t _tile_width, uint32_t _power_cycle_energy)
        : layer_dims(_layer_dims)
        , tile_channel(_tile_channel)
        , tile_width(_tile_width)
        , tile_channel_largest_local_minimum(_tile_channel)
        , tile_width_largest_local_minimum(_tile_width)
        , power_cycle_energy(_power_cycle_energy)
    {
        tile_channel_largest_local_minimum = nearest_value(ParameterDimension::TileChannel, tile_channel, /*not_larger_than=*/true);
    }
    uint32_t calc(uint8_t dim_idx, uint16_t dim_value) const;
    uint16_t nearest_value(uint8_t dim_idx, uint16_t dim_value, bool not_larger_than) const;

    enum ParameterDimension {
        TileChannel,
        TileWidth,
    };
private:
    const FcLayerDimensions& layer_dims;
    uint16_t tile_channel;
    uint16_t tile_width;
    uint16_t tile_channel_largest_local_minimum;
    uint16_t tile_width_largest_local_minimum;
    uint32_t power_cycle_energy;
};

void update_progress_indicator_fc(const Node* node, NodeFlags* flags, const NodeFlags* orig_flags, const FcLayerDimensions& layer_dims, uint32_t first_unfinished_value_offset);
