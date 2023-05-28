#pragma once

#include <cstdint>

struct ConvNodeFlags {
    uint16_t input_tile_c;
    uint16_t output_tile_c;
    uint8_t pads[4];
    uint8_t kernel_shape[2];
    uint8_t strides[2];
    uint8_t group;
    uint8_t dummy;
    uint16_t pState_len;
};

struct MaxPoolFlags {
    uint8_t kernel_shape[2];
    uint8_t strides[2];
    uint8_t ceil;
    uint8_t nhwc2nchw;
};

struct GemmNodeFlags {
    uint16_t tile_channel;
    uint16_t tile_width;
    uint16_t pState_len;
};

struct GemmMergeNodeFlags {
    uint16_t tile_length;
};

struct SqueezeNodeFlags {
    // a bitmap for axes to squeeze/unsqueeze
    uint8_t axes;
};

struct ConcatNodeFlags {
    int8_t axis;
};

#define NODE_FLAGS_SIZE 16

struct NodeFlags {
    union {
        struct ConvNodeFlags conv;
        struct MaxPoolFlags maxpool;
        struct GemmNodeFlags gemm;
        struct GemmMergeNodeFlags gemmmerge;
        struct SqueezeNodeFlags squeeze;
        struct ConcatNodeFlags concat;
        uint8_t as_bytes[NODE_FLAGS_SIZE];
    };
    // `canary` contains some non-zero value for detecting whether data are already in VM or not
    uint8_t canary;
    uint8_t version;
};

static_assert(sizeof(struct NodeFlags) == NODE_FLAGS_SIZE + 2, "Unexpected size for NodeFlags");

struct Footprint {
    uint16_t value;
    uint8_t version;
    uint8_t dummy;
};

struct InferenceStats {
    uint32_t last_progress_indicator;
    uint32_t power_cycle_energy;
    uint8_t dummy[3];
    uint8_t version;
};

#define N_PERSISTENT_COUNTERS 1
struct Counters {
#if !ENABLE_DEMO_COUNTERS
    uint32_t power_counters;
    uint32_t macs;

    uint32_t embedding;
    uint32_t stripping;
    uint32_t overflow_handling;

    uint32_t state_query;
    uint32_t table_updates;
    uint32_t table_preservation;
    uint32_t table_loading;

    uint32_t progress_seeking;

    uint32_t memory_layout;

    uint32_t data_loading;

    uint32_t embedded_values;

    uint32_t dma_invocations;
    uint32_t dma_bytes;
    uint32_t dma_vm_to_vm;
    uint32_t nvm_read_job_outputs;
    uint32_t nvm_read_parameters;
    uint32_t nvm_read_shadow_data;
    uint32_t nvm_read_model;
    uint32_t nvm_write_shadow_data;
    uint32_t nvm_write_model;
#endif
    uint32_t nvm_write_linear_jobs;
    uint32_t nvm_write_non_linear_jobs;
    uint32_t nvm_write_footprints;

    // persistent counters
    uint32_t total_jobs;
};
