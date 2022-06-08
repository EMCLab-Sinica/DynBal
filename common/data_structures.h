#include <cstdint>

struct ConvNodeFlags {
    uint16_t input_tile_c;
    uint16_t output_tile_c;
    uint8_t pads[4];
    uint8_t kernel_shape[2];
    uint8_t strides[2];
};

struct MaxPoolFlags {
    uint8_t kernel_shape[2];
    uint8_t strides[2];
    uint8_t ceil;
    uint8_t nhwc2nchw;
};

struct GemmNodeFlags {
    uint16_t tile_channel;
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

#define NODE_FLAGS_SIZE 12

union NodeFlags {
    struct ConvNodeFlags conv;
    struct MaxPoolFlags maxpool;
    struct GemmNodeFlags gemm;
    struct GemmMergeNodeFlags gemmmerge;
    struct SqueezeNodeFlags squeeze;
    struct ConcatNodeFlags concat;
    uint8_t as_bytes[NODE_FLAGS_SIZE];
};

static_assert(sizeof(union NodeFlags) == NODE_FLAGS_SIZE, "Unexpected size for NodeFlags");
