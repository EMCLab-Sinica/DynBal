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
};

struct MaxPoolFlags {
    uint8_t kernel_shape[2];
    uint8_t strides[2];
    uint8_t ceil;
    uint8_t nhwc2nchw;
};

struct GemmNodeFlags {
    uint16_t tile_channel;
    uint16_t op_filters;
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

#define NODE_FLAGS_SIZE 14

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
