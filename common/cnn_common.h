#pragma once

#include <cstddef> /* size_t, see https://stackoverflow.com/a/26413264 */
#include <cstdint>
#include "data.h"
#include "data_structures.h"

/**********************************
 *        Data structures         *
 **********************************/

#if defined(__GNUC__) || defined(__clang__)
static_assert(true, "Dummy declaration to workaround https://github.com/clangd/clangd/issues/1167");
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wpadded"
#endif

typedef struct Node {
    char name[NODE_NAME_LEN];
    char output_name[NODE_NAME_LEN];
    uint16_t inputs_len;
    int16_t inputs[NUM_INPUTS];
    uint16_t max_output_id;
    uint16_t op_type;
    const NodeFlags orig_flags;
#if HAWAII
    struct Footprint {
        uint16_t value;
        uint8_t version;
        uint8_t dummy;
    } footprint[2];
#endif
} Node;

static_assert(sizeof(Node) == NODE_NAME_LEN * 2 + 22 + NUM_INPUTS * 2 + HAWAII * 8, "Unexpected size for Node");

struct Scale {
    int16_t fract;
    uint8_t shift;
    uint8_t dummy;

    bool operator>(const Scale& other) const;
    Scale operator*(const Scale& other) const;
    Scale operator/(const Scale& other) const;
    bool operator!=(const Scale& other) const;
    float toFloat() const;
};

/* ParameterInfo may indicate data from the model (parameters) or intermediate values */
typedef struct ParameterInfo {
    uint32_t params_offset;
    uint32_t params_len;  /* in bytes */
    /* A flag to indicate where the data are. Possible values are SLOT_TEST_SET,
     * SLOT_PARAMETERS and a value in [0, NUM_SLOTS-1].
     */
    uint8_t slot;
    uint8_t param_flags;
    // uint8_t is not enough. For example, fully connected layer in MNIST has dims 256x1
    uint16_t dims[4];
    Scale scale;
    uint16_t parameter_info_idx; // must be the last member of this struct
} ParameterInfo;

static_assert(sizeof(ParameterInfo) == 24, "Unexpected size for ParameterInfo");

typedef struct SlotInfo {
#if INDIRECT_RECOVERY
    int8_t state_bit;
    uint8_t n_turning_points;
    uint16_t turning_points[TURNING_POINTS_LEN];
#endif
    int16_t user;
} SlotInfo;

typedef struct Model {
    uint16_t running;
    uint16_t run_counter;
    uint16_t layer_idx;
    SlotInfo slots_info[NUM_SLOTS];
    uint8_t dummy;
    uint8_t version; // must be the last field in this struct
} Model;

static_assert(sizeof(Model) == 8 + NUM_SLOTS * (2 + INDIRECT_RECOVERY * (2 + TURNING_POINTS_LEN * 2)), "Unexpected size for Model");

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

/**********************************
 *          Global data           *
 **********************************/
extern ParameterInfo intermediate_parameters_info_vm[MODEL_NODES_LEN];
extern uint16_t sample_idx;

/**********************************
 *         The entry point        *
 **********************************/
uint8_t run_cnn_tests(uint16_t n_samples);

/**********************************
 *          Miscellaneous         *
 **********************************/

/* MSP430 SDK already defines MIN, which means minutes */
#define MIN_VAL(x, y) ((x) < (y) ? (x) : (y))
#define MAX_VAL(x, y) ((x) > (y) ? (x) : (y))
// XXX: MSP432 driverlib requires DMA transfer size to be <= 1024. However,
// transfer size < 1024 may be broken as well - copying 1024 items works,
// copying 512 items works, copy a small number of items (e.g., 6, 10, ...)
// works, and copying 626 items (in ConvMerge of conv2 in MNIST) DOES NOT
// WORK (!?).
#define LIMIT_DMA_SIZE(x) MIN_VAL(512, x)

/**********************************
 * Helpers for the model & nodes  *
 **********************************/
const uint8_t* get_param_base_pointer(const ParameterInfo *param, uint32_t *limit_p);
int16_t get_q15_param(Model* model, const ParameterInfo *param, uint16_t offset_in_word);
void put_q15_param(ParameterInfo *param, uint16_t offset_in_word, int16_t val, bool is_linear);
int64_t get_int64_param(const ParameterInfo *param, size_t i);
uint16_t get_next_slot(Model *model, const ParameterInfo *param);
const ParameterInfo* get_parameter_info(uint16_t i);
const Node* get_node(size_t i);
const Node* get_node(const ParameterInfo* param);
SlotInfo * get_slot_info(Model* model, uint8_t i);
void my_memcpy_from_param(Model* model, void *dest, const ParameterInfo *param, uint16_t offset_in_word, size_t n);

/**********************************
 *       Operation handlers       *
 **********************************/
typedef void (*handler)(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node* node, NodeFlags* node_flags, const NodeFlags* orig_node_flags);
typedef void (*allocator)(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node* node, NodeFlags* node_flags, const NodeFlags* orig_node_flags);
// below are defined in ops.c
extern const handler handlers[];
extern const allocator allocators[];
