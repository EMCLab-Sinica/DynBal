#pragma once

#include <cstdint>

#define Fixed 0
#define DynBal 1
#define Exhaustive 2
#define RuntimeConfiguration DynBal

const uint32_t NVM_RELATIVE_WRITE_COST = 1; // the ratio of NVM write cost and NVM read cost

class UsageSpan {
public:
    virtual uint32_t calc(uint8_t dim_idx, uint16_t dim_value) const = 0;
    virtual uint16_t nearest_value(uint8_t dim_idx, uint16_t dim_value) const = 0;
};

uint16_t convex_search(const UsageSpan* usage_span, uint8_t dim_idx, const uint16_t value_ranges[][2]);

#if RuntimeConfiguration == DynBal
struct InferenceStats {
    uint32_t last_progress_indicator;
    uint32_t power_cycle_energy;
    uint8_t dummy[3];
    uint8_t version;
};
extern InferenceStats inference_stats_vm[2];

enum class InferenceStatsOpType {
    Conv,
    FC,
};
void commit_inference_stats(InferenceStatsOpType op_type);
InferenceStats* load_inference_stats_from_nvm(InferenceStatsOpType op_type);
#endif

#if RuntimeConfiguration == Exhaustive
struct ExhaustiveLookupTable {
    uint16_t node_idx;
    uint16_t conv_output_tile_c;
};
#endif
