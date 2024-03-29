#include <cstddef>
#include <cstdint>
#include <cinttypes> // for PRId32
#include "cnn_common.h"
#include "conv.h"
#include "counters.h"
#include "data.h"
#include "my_debug.h"
#include "op_utils.h"
#include "intermittent-cnn.h"
#include "my_dsplib.h"
#include "platform.h"
#include "dynbal.h"
#include "dynbal-conv.h"
#include "layers.h"

#define ON_DEMAN_TILE_LOADING 1

/* Better to not use macros
 * https://stackoverflow.com/a/3437484/3786245
 */
static inline int16_t int16_min(int16_t a, int16_t b) {
    return a < b ? a : b;
}

static inline int16_t int16_max(int16_t a, int16_t b) {
    return a > b ? a : b;
}

#define CONV_TASK_FLAG_PROCESSED_FILTERS_BASE 2
typedef struct ConvTaskParams {
    Model* model;
    const ParameterInfo *conv_input;
    const ParameterInfo *conv_filter;
    const ParameterInfo *conv_bias;
    ParameterInfo *output;
    NodeFlags* flags;
    const NodeFlags* orig_flags;

    /* aux vars remaining constant for a conv layer */
    ConvLayerDimensions layer_dims;
    uint16_t pState_len;
    uint8_t group;
    uint16_t input_tile_c_offset;
    uint16_t input_tile_c_index;
    uint16_t cur_input_tile_c;
    uint16_t cur_filter_tile_c;
    uint16_t n_tiles_c;
    uint16_t dest_offset;
    uint16_t filter_offset;
    uint8_t truncated;
#if INDIRECT_RECOVERY
    int16_t old_output_offset ;
    uint8_t turning_point_idx;
    uint16_t next_turning_point;
    SlotInfo* cur_slot_info;
#endif
#if JAPARI
    uint8_t conv_input_has_footprints;
    uint16_t input_tile_c_offset_with_footprints;
    uint8_t force_align_footprints;
#endif
#if STATEFUL
    uint8_t output_padding;
#endif

    uint16_t filter_idx;
    uint16_t filter_tile_index;
    // (h, w) for left-top corner of each input window
    int16_t input_h;
    int16_t input_w;
    int16_t input_h_first, input_h_last;
    int16_t input_w_first, input_w_last;
    int16_t *filter_buffer_addr;
    int16_t cached_filter_idx;
    uint16_t cached_input_tile_c_offset;

#if DYNBAL_REPORT_PARAMETERS
    uint32_t first_unfinished_job_idx;
    uint8_t reported;
#endif
} ConvTaskParams;

static ConvTaskParams conv_params_obj;

// for group convolution
int16_t* const biases = op_buffer;

#if INDIRECT_RECOVERY
static void flip_filter_state_bits(ConvTaskParams *conv_params, uint16_t n_filters) {
#if STATEFUL
    int16_t *to_flip_state_bits = nullptr;
    if (conv_params->group == 1) {
        to_flip_state_bits = conv_params->filter_buffer_addr + n_filters * (conv_params->filter_offset - 1);
    } else {
        to_flip_state_bits = biases;
    }
#if ENABLE_COUNTERS && !ENABLE_DEMO_COUNTERS
    add_counter(offsetof(Counters, embedded_values), n_filters);
#endif
    // need negating filter value here as it will be multiplied with _Q15(-1.0), or -32768
    int8_t state_multiplier = (conv_params->group == 1) ? -1 : 1;
    for (uint16_t idx = BATCH_SIZE - 1; idx < n_filters; idx += BATCH_SIZE) {
        if (get_value_state_bit(to_flip_state_bits[idx]) != state_multiplier * get_value_state_bit(state_offsets[idx])) {
            my_printf_debug("Flipping state bit in filter %d" NEWLINE, idx);
            to_flip_state_bits[idx] += 0x8000;
        }
    }
#endif
#if JAPARI
    int16_t *to_flip_state_bits = conv_params->filter_buffer_addr + n_filters * (conv_params->filter_offset - 1);
    for (uint16_t idx = BATCH_SIZE; idx < n_filters; idx += BATCH_SIZE + 1) {
        to_flip_state_bits[idx] = (state_offsets[idx] == 0x4000) ? -1 : 1;
    }
#endif
}
#endif

static void convTask(int16_t cur_input_h, const ConvLayerDimensions* layer_dims, ConvTaskParams *conv_params) {
    int16_t output_tile_c = conv_params->flags->conv.output_tile_c;
    int16_t cur_output_tile_c;
    if (conv_params->filter_idx >= layer_dims->N_FILTERS - layer_dims->N_FILTERS % output_tile_c) {
        cur_output_tile_c = layer_dims->N_FILTERS - conv_params->filter_idx;
    } else {
        cur_output_tile_c = output_tile_c - conv_params->filter_idx % output_tile_c;
    }
    my_printf_debug("cur_output_tile_c = %d" NEWLINE, cur_output_tile_c);
    MY_ASSERT(cur_output_tile_c > 0);

    int16_t n_filters = cur_output_tile_c;
    int16_t values_to_preserve = n_filters;
    int16_t channel_offset_c = conv_params->filter_idx;
#if JAPARI
    start_cpu_counter(offsetof(Counters, embedding));
    values_to_preserve = extend_for_footprints(n_filters, conv_params->force_align_footprints);
    n_filters = padding_for_lea(values_to_preserve);
    channel_offset_c = extend_for_footprints(channel_offset_c);
    stop_cpu_counter();
#endif
#if STATEFUL
    start_cpu_counter(offsetof(Counters, memory_layout));
    if (conv_params->output_padding) {
        values_to_preserve += conv_params->output_padding;
        n_filters = padding_for_lea(values_to_preserve);
    }
    stop_cpu_counter();
#endif
    uint16_t output_h = (cur_input_h - conv_params->input_h_first) / layer_dims->STRIDE_H,
             output_w = (conv_params->input_w - conv_params->input_w_first) / layer_dims->STRIDE_W;
    // use NWHC so that output is written continuously on the address space
    uint16_t cur_output_data_offset =
             layer_dims->OUTPUT_W * layer_dims->OUTPUT_H * (conv_params->input_tile_c_index * layer_dims->OUTPUT_CHANNEL) +  // n
             output_w * layer_dims->OUTPUT_H * layer_dims->OUTPUT_CHANNEL +                                                   // w
             output_h * layer_dims->OUTPUT_CHANNEL +                                                                           // h
             channel_offset_c;                                                                                                  // c

#if INDIRECT_RECOVERY
    start_cpu_counter(offsetof(Counters, embedding));
    fill_state_offsets(cur_output_data_offset, n_filters, &conv_params->old_output_offset, &conv_params->turning_point_idx, &conv_params->next_turning_point, conv_params->cur_slot_info);
    my_printf_debug("State offsets" NEWLINE);
    dump_matrix_debug(state_offsets, n_filters, ValueInfo(conv_params->conv_input, nullptr), false);
    stop_cpu_counter();
#endif
    int16_t * const matrix_mpy_results = lea_buffer + LEA_BUFFER_SIZE - OUTPUT_LEN - conv_params->pState_len;

    /* copy filter data */
    if (conv_params->cached_filter_idx != conv_params->filter_idx || conv_params->cached_input_tile_c_offset != conv_params->input_tile_c_offset) {
        int16_t *filter_tmp = matrix_mpy_results - (conv_params->filter_offset + 1) / 2 * 2; // before transpose
        conv_params->filter_buffer_addr = filter_tmp - conv_params->filter_offset * n_filters;
        my_fill_q15(0, conv_params->filter_buffer_addr, conv_params->filter_offset * n_filters);

        uint16_t fill_length = conv_params->filter_offset;
        my_fill_q15(0, filter_tmp, fill_length);
        uint16_t filter_len = layer_dims->kH * layer_dims->kW * layer_dims->CHANNEL;
        for (uint16_t idx = 0; idx < cur_output_tile_c; idx++) {
            uint16_t filter_src_offset = (conv_params->filter_idx + idx) * filter_len;
            my_printf_debug("Copying filter %d" NEWLINE, conv_params->filter_idx + idx);
            if (conv_params->cur_filter_tile_c == layer_dims->CHANNEL && layer_dims->kW * layer_dims->CHANNEL == conv_params->dest_offset) {
                uint16_t cur_filter_src_offset = filter_src_offset + conv_params->input_tile_c_offset;
                uint16_t buffer_size = conv_params->cur_filter_tile_c * layer_dims->kH * layer_dims->kW;
                my_memcpy_from_param(conv_params->model, filter_tmp, conv_params->conv_filter, cur_filter_src_offset, sizeof(int16_t) * buffer_size);
                my_printf_debug("[%d, %d) => lea_buffer + [%ld, %ld)" NEWLINE,
                                cur_filter_src_offset, cur_filter_src_offset + buffer_size,
                                filter_tmp - lea_buffer, filter_tmp + buffer_size - lea_buffer);
            } else {
                for (uint16_t h = 0; h < layer_dims->kH; h++) {
                    int16_t *filter_dest_ptr = filter_tmp + h * conv_params->dest_offset;
                    uint16_t cur_filter_src_offset = filter_src_offset + h * layer_dims->kW * layer_dims->CHANNEL + conv_params->input_tile_c_offset;
                    if (conv_params->cur_filter_tile_c == layer_dims->CHANNEL) {
                        uint16_t buffer_size = conv_params->cur_filter_tile_c * layer_dims->kW;
                        my_printf_debug("[%d, %d) => lea_buffer + [%ld, %ld)" NEWLINE,
                                        cur_filter_src_offset, cur_filter_src_offset + buffer_size,
                                        filter_dest_ptr - lea_buffer, filter_dest_ptr + buffer_size - lea_buffer);
                        my_memcpy_from_param(conv_params->model, filter_dest_ptr, conv_params->conv_filter, cur_filter_src_offset, sizeof(int16_t) * buffer_size);
                    } else {
                        for (uint16_t w = 0; w < layer_dims->kW; w++) {
                            my_printf_debug("[%d, %d) => lea_buffer + [%ld, %ld)" NEWLINE,
                                            cur_filter_src_offset, cur_filter_src_offset + conv_params->cur_filter_tile_c,
                                            filter_dest_ptr - lea_buffer, filter_dest_ptr + conv_params->cur_filter_tile_c - lea_buffer);
                            my_memcpy_from_param(conv_params->model, filter_dest_ptr, conv_params->conv_filter, cur_filter_src_offset, sizeof(int16_t) * conv_params->cur_filter_tile_c);
                            filter_dest_ptr += conv_params->cur_filter_tile_c;
                            cur_filter_src_offset += layer_dims->CHANNEL;
                        }
                    }
                }
            }
            int16_t last_elem = 0;
#if STATEFUL
            start_cpu_counter(offsetof(Counters, embedding));
            if (conv_params->conv_input->slot == SLOT_TEST_SET) {
                my_scale_q15(filter_tmp, 0x4000, 0, filter_tmp, conv_params->filter_offset);
            }
            bool has_state = offset_has_state(cur_output_data_offset + idx);
            if (has_state) {
                my_printf_debug("Adding state bit for newly loaded filter idx=%d" NEWLINE, idx);
                last_elem = state_offsets[idx];
#if ENABLE_COUNTERS && !ENABLE_DEMO_COUNTERS
                add_counter(offsetof(Counters, embedded_values), 1);
#endif
            }
            stop_cpu_counter();
            if (!has_state)
#endif
                if (conv_params->group == 1) {
                    // XXX: why is this needed? Should already be zero with my_fill_q15 above
                    last_elem = 0;
                }
            int16_t bias_val = 0;
            if (conv_params->input_tile_c_index == 0) {
                if (conv_params->conv_bias) {
                    // convert int16_t to int32_t first as on MSP430, registers are 20 bit while there are only 16 bits when int16_t is converted to uint16_t
                    // If the dividend is negative, the quotient is wrong
                    bias_val = static_cast<int32_t>(get_q15_param(conv_params->model, conv_params->conv_bias, conv_params->filter_idx + idx)) / conv_params->conv_input->scale.toFloat();
                }
#if STATEFUL
                start_cpu_counter(offsetof(Counters, embedding));
                if (conv_params->conv_input->slot == SLOT_TEST_SET) {
                    bias_val /= 2;
                }
                stop_cpu_counter();
#endif
            }
            last_elem += bias_val;
            if (conv_params->group == 1) {
                filter_tmp[conv_params->filter_offset - 1] = -last_elem;
            } else {
                biases[idx] = last_elem;
            }

            uint16_t channel = idx;
#if JAPARI
            start_cpu_counter(offsetof(Counters, memory_layout));
            channel += channel / BATCH_SIZE;
            stop_cpu_counter();
#endif
            my_interleave_q15(filter_tmp, channel, n_filters, conv_params->filter_buffer_addr, conv_params->filter_offset);
        }

#if JAPARI
        start_cpu_counter(offsetof(Counters, embedding));
        int16_t* footprint_channels_ptr = conv_params->filter_buffer_addr + n_filters * (conv_params->filter_offset - 1);
        for (int16_t idx = BATCH_SIZE; idx < n_filters; idx += BATCH_SIZE + 1) {
            *(footprint_channels_ptr + idx) = (state_offsets[idx] == 0x4000 ? -1 : 1);
        }
        stop_cpu_counter();
#endif

#if STATEFUL
        start_cpu_counter(offsetof(Counters, memory_layout));
        if (conv_params->output_padding) {
            conv_params->filter_buffer_addr[n_filters * conv_params->filter_offset - 1] = -state_offsets[n_filters - 1];
        }
        stop_cpu_counter();
#endif

        conv_params->cached_filter_idx = conv_params->filter_idx;
        conv_params->cached_input_tile_c_offset = conv_params->input_tile_c_offset;
    } else {
#if INDIRECT_RECOVERY
        start_cpu_counter(offsetof(Counters, embedding));
        flip_filter_state_bits(conv_params, n_filters);
        stop_cpu_counter();
#endif
    }

    int16_t *filter_buffer_addr = conv_params->filter_buffer_addr;

    int16_t *input_buffer_addr = lea_buffer + (cur_input_h-conv_params->input_h) * conv_params->dest_offset * conv_params->group;

    uint16_t A_rows, A_cols, B_rows, B_cols;
    A_rows = 1;
    A_cols = conv_params->filter_offset * conv_params->group;
    B_rows = conv_params->filter_offset;
    B_cols = n_filters;
    MY_ASSERT(input_buffer_addr + A_rows * A_cols <= filter_buffer_addr);
    if (conv_params->group == 1) {
        MY_ASSERT(A_rows * B_cols <= OUTPUT_LEN);
        my_matrix_mpy_q15(A_rows, A_cols, B_rows, B_cols, input_buffer_addr, filter_buffer_addr, matrix_mpy_results,
                          conv_params->output, cur_output_data_offset, values_to_preserve, conv_params->pState_len);
    } else {
        MY_ASSERT(B_rows * B_cols <= OUTPUT_LEN);
        if (n_filters == conv_params->group) {
            int16_t* cur_matrix_mpy_results = matrix_mpy_results + n_filters;
            my_vector_mult_q15(input_buffer_addr, filter_buffer_addr, matrix_mpy_results, B_rows * B_cols);
            for (uint16_t row = 1; row < B_rows; row++) {
                my_add_q15(matrix_mpy_results, cur_matrix_mpy_results, matrix_mpy_results, conv_params->group);
                cur_matrix_mpy_results += n_filters;
            }
        } else {
            int16_t *cur_input_buffer_addr = input_buffer_addr + (conv_params->group - n_filters),
                    *cur_filter_buffer_addr = filter_buffer_addr,
                    *cur_matrix_mpy_results = matrix_mpy_results;
            for (uint16_t row = 0; row < B_rows; row++) {
                my_vector_mult_q15(cur_input_buffer_addr, cur_filter_buffer_addr, cur_matrix_mpy_results, n_filters);
                if (row) {
                    my_add_q15(matrix_mpy_results, cur_matrix_mpy_results, matrix_mpy_results, n_filters);
                }
                cur_input_buffer_addr += conv_params->group;
                cur_filter_buffer_addr += n_filters;
                cur_matrix_mpy_results += n_filters;
            }
        }
        my_add_q15(matrix_mpy_results, biases, matrix_mpy_results, n_filters);
        my_memcpy_to_param(conv_params->output, cur_output_data_offset, matrix_mpy_results, n_filters * sizeof(int16_t), 0, true);
    }

    /* START dump data */
    my_printf_debug("input_h=%d" NEWLINE, cur_input_h);
    my_printf_debug("filter_idx=");
#if MY_DEBUG >= MY_DEBUG_VERBOSE
    for (uint16_t idx = 0; idx < cur_output_tile_c; idx++) {
        my_printf_debug("%d ", conv_params->filter_idx + idx);
        MY_ASSERT(conv_params->filter_idx + idx < layer_dims->N_FILTERS);
    }
#endif
    my_printf_debug("output_h=%d output_w=%d" NEWLINE, output_h, output_w);

    my_printf_debug("input" NEWLINE);
    dump_matrix_debug(input_buffer_addr, A_rows, A_cols, ValueInfo(conv_params->conv_input, nullptr), false);
    my_printf_debug("filter_buffer_addr = lea_buffer + LEA_BUFFER_SIZE - OUTPUT_LEN - pState_len - %d" NEWLINE,
                    static_cast<int>(lea_buffer + LEA_BUFFER_SIZE - OUTPUT_LEN - conv_params->pState_len - filter_buffer_addr));
    my_printf_debug("filter" NEWLINE);
    dump_matrix_debug(filter_buffer_addr, B_rows, B_cols, ValueInfo(conv_params->conv_filter, nullptr), false);
    if (conv_params->group != 1) {
        my_printf_debug("biases" NEWLINE);
        dump_matrix_debug(biases, 1, n_filters, ValueInfo(conv_params->conv_filter, nullptr), false);
    }

    my_printf_debug("matrix_mpy_results" NEWLINE);
    dump_matrix_debug(matrix_mpy_results, A_rows, B_cols, ValueInfo(conv_params->output));
    my_printf_debug(NEWLINE);

    compare_vm_nvm(matrix_mpy_results, conv_params->model, conv_params->output, cur_output_data_offset, values_to_preserve);
    /* END dump data */

    my_printf_debug("output_data offset = %d" NEWLINE, cur_output_data_offset);

    MY_ASSERT(cur_output_data_offset + n_filters < INTERMEDIATE_VALUES_SIZE * NUM_SLOTS);

#if HAWAII
    hawaii_record_footprints(conv_params->model, values_to_preserve);
#endif
}

static inline uint16_t load_input_vector(uint32_t src_addr, int16_t* dest_addr, uint16_t len, const ConvTaskParams* conv_params) {
    my_printf_debug("Load %d IFM values [%d, %d) => lea_buffer + [%ld, %ld) ",
                    len, src_addr, static_cast<int>(src_addr + len), dest_addr - lea_buffer, dest_addr + len - lea_buffer);
    int16_t* memcpy_dest_addr = nullptr;
    uint16_t loaded_len = 0;

    MY_ASSERT(len != 0);

#if JAPARI
    if (conv_params->conv_input_has_footprints) {
        MY_ASSERT(len <= INPUT_BUFFER_WITH_FOOTPRINTS_LEN);
        memcpy_dest_addr = input_buffer_with_footprints;
    } else
#endif
    {
        memcpy_dest_addr = dest_addr;
        loaded_len = len;
    }
    my_memcpy_from_param(
        conv_params->model, memcpy_dest_addr,
        conv_params->conv_input, src_addr,
        len * sizeof(int16_t));
#if JAPARI
    start_cpu_counter(offsetof(Counters, stripping));
    if (conv_params->conv_input_has_footprints) {
        // Use nested loops as skipping footprints by `% (BATCH_SIZE)` is quite slow on boards
        int16_t *dest_ptr = dest_addr,
                *src_ptr = input_buffer_with_footprints;
        for (uint16_t src_idx = 0; src_idx < len; src_idx += (BATCH_SIZE + 1)) {
            for (uint8_t batch_offset = 0; batch_offset < BATCH_SIZE; batch_offset++) {
                *dest_ptr = *src_ptr;
                dest_ptr++;
                src_ptr++;
            }
            src_ptr++; // skipping footprints
        }
        loaded_len = dest_ptr - dest_addr;
    }
    stop_cpu_counter();
#endif

#if MY_DEBUG >= MY_DEBUG_VERBOSE
    for (uint16_t idx = 0; idx < loaded_len; idx++) {
        my_printf_debug("%d ", dest_addr[idx]);
    }
    my_printf_debug(NEWLINE);
#endif
    return loaded_len;
}

static void load_ifm_tile_row(Model* model, const ConvLayerDimensions* layer_dims, const ConvTaskParams* conv_params, int32_t h_start, int32_t h_end) {
    /* copy input data, row by row */

    /* int32_t instead of int16_t as TI's compiler cannot handle negative
     * offsets correctly. The expression `ptr + (int16_t)(-2)` is
     * compiled as:
     * 1. -2 is represented as 0x00FFFE (general registers are 24-bit long).
     *    Assume this value is stored in R11.
     * 2. RLAM.A #1,R11  # multiply by 2 to transform the offset for int16_t
     *    to the difference of addresses.
     * In step 2, R11 becomes 0x01FFFC, while it should be -4, or 0x00FFFC,
     * and thus the resultant address is offset by 0x10000.
     */
    int32_t w_start = int16_max(0, conv_params->input_w),
            w_end   = int16_min(conv_params->input_w+layer_dims->kW-1, layer_dims->W-1);
    int16_t *dest;

    dest = lea_buffer;

    dest += (h_start-conv_params->input_h) * conv_params->dest_offset * conv_params->group;

    my_printf_debug("h_start=%" PRId32 " ", h_start);

    uint16_t cur_input_tile_c = conv_params->cur_input_tile_c;
    uint8_t im2col_channel_offset = cur_input_tile_c;
    my_printf_debug("Copying row to lea_buffer + %d" NEWLINE,
                    static_cast<int>(dest - lea_buffer));
    uint16_t cur_input_channel = layer_dims->CHANNEL;
#if JAPARI
    start_cpu_counter(offsetof(Counters, embedding));
    if (conv_params->conv_input_has_footprints) {
        cur_input_tile_c = extend_for_footprints(cur_input_tile_c);
        cur_input_channel = extend_for_footprints(cur_input_channel);
    }
    stop_cpu_counter();
#endif
    int16_t input_src_offset;
    uint8_t transposed = conv_params->conv_input->param_flags & TRANSPOSED;
    if (transposed) {
        input_src_offset = (w_start * layer_dims->H + h_start) * cur_input_channel * conv_params->group;
    } else {
        input_src_offset = (h_start * layer_dims->W + w_start) * cur_input_channel * conv_params->group;
    }
#if JAPARI
    input_src_offset += conv_params->input_tile_c_offset_with_footprints;
#else
    input_src_offset += conv_params->input_tile_c_offset;
#endif
#if INDIRECT_RECOVERY
    dump_turning_points_debug(model, conv_params->conv_input);
#endif
    int16_t input_src_vertical_offset;
    if (transposed) {
        input_src_vertical_offset = cur_input_channel * conv_params->group;
    } else {
        input_src_vertical_offset = layer_dims->W * cur_input_channel * conv_params->group;
    }
    uint16_t input_row_len = (w_end - w_start + 1) * cur_input_tile_c * conv_params->group;

    if (input_src_vertical_offset == conv_params->dest_offset * conv_params->group) {
        // XXX: make them compatible
        MY_ASSERT(!ON_DEMAN_TILE_LOADING);
        int16_t *dest_addr = dest + (w_start-conv_params->input_w) * im2col_channel_offset * conv_params->group;
        uint16_t input_tile_len = (h_end-h_start+1)*input_row_len;
        load_input_vector(input_src_offset, dest_addr, input_tile_len, conv_params);
#if STATEFUL
        start_cpu_counter(offsetof(Counters, stripping));
        if (conv_params->conv_input->slot != SLOT_TEST_SET) {
            MY_ASSERT(cur_input_tile_c % BATCH_SIZE == 0);
            for (int16_t *dest_ptr = dest_addr + BATCH_SIZE - 1; dest_ptr < dest_addr + input_tile_len; dest_ptr += BATCH_SIZE) {
                strip_state(dest_ptr);
            }
        }
        stop_cpu_counter();
#endif
    } else {

    for (int32_t h = h_start; h <= h_end; h++) {
        int16_t *dest_addr = dest + (w_start-conv_params->input_w) * im2col_channel_offset * conv_params->group;
#if STATEFUL
        int16_t *orig_dest_addr = dest_addr;
#endif
        uint32_t src_addr = input_src_offset;
        if ((cur_input_tile_c == cur_input_channel) && !transposed) {
            load_input_vector(src_addr, dest_addr, input_row_len, conv_params);
        } else {
            for (int32_t w = w_start; w <= w_end; w++) {
                load_input_vector(src_addr, dest_addr, cur_input_tile_c * conv_params->group, conv_params);
                dest_addr += im2col_channel_offset * conv_params->group;
                if (transposed) {
                    src_addr += layer_dims->H * cur_input_channel * conv_params->group;
                } else {
                    src_addr += cur_input_channel * conv_params->group;
                }
            }
        }

#if STATEFUL
        start_cpu_counter(offsetof(Counters, stripping));
        if (conv_params->conv_input->slot != SLOT_TEST_SET) {
            // stripping states inside the h loop is faster as biases multipliers can be skipped
            int16_t *input_row_end = orig_dest_addr + input_row_len;
            // if input_tile_c is smaller than BATCH_SIZE, state bits are not always at offset BATCH_SIZE - 1
            my_printf_debug("Using a loop for stripping state bits" NEWLINE);
            MY_ASSERT(cur_input_tile_c % BATCH_SIZE == 0 || BATCH_SIZE % cur_input_tile_c == 0);
            if (cur_input_tile_c % BATCH_SIZE == 0) {
                for (int16_t *dest_ptr = orig_dest_addr + BATCH_SIZE - 1; dest_ptr < input_row_end; dest_ptr += BATCH_SIZE) {
                    strip_state(dest_ptr);
                }
            } else {
                int16_t offset = BATCH_SIZE - 1 - src_addr % BATCH_SIZE;
                if (offset < cur_input_tile_c) {
                    for (; offset < input_row_len; offset += cur_input_tile_c) {
                        strip_state(orig_dest_addr + offset);
                    }
                }
            }
        }
        stop_cpu_counter();
#endif
        dest += conv_params->dest_offset * conv_params->group;
        input_src_offset += input_src_vertical_offset;
    }

    }
}

static uint16_t handle_conv_inner_loop(Model *model, const ConvLayerDimensions* layer_dims, ConvTaskParams *conv_params) {
    int8_t field_size = (layer_dims->kH - 1) / 2;
    int16_t max_n_filters = conv_params->flags->conv.output_tile_c;
#if JAPARI
    start_cpu_counter(offsetof(Counters, memory_layout));
    max_n_filters *= 2;
    stop_cpu_counter();
#endif
    // 1 additional filters for values before transpose
    uint16_t inputs_buffer_end = LEA_BUFFER_SIZE - OUTPUT_LEN - conv_params->pState_len - (max_n_filters + 1) * conv_params->filter_offset;
    uint16_t tile_h = MIN_VAL(inputs_buffer_end / (conv_params->group * conv_params->dest_offset) - 2 * field_size, layer_dims->H);
#if DYNBAL_REPORT_PARAMETERS
    if (conv_params->first_unfinished_job_idx == 0 && !conv_params->reported) {
        my_printf("%d" NEWLINE, tile_h);
        conv_params->reported = 1;
    }
#endif
    uint16_t inputs_len = (tile_h + 2 * field_size) * (conv_params->group * conv_params->dest_offset);
    MY_ASSERT(inputs_len < LEA_BUFFER_SIZE - OUTPUT_LEN - conv_params->pState_len); // make sure no overflow occurs in the previous line

    my_printf_debug("Reinitialize input buffer" NEWLINE "tile_h = %d, inputs_len = %d" NEWLINE, tile_h, inputs_len);

    my_fill_q15(0, lea_buffer, inputs_len);
    // XXX: write multipliers on demand?
    if (conv_params->group == 1) {
        uint16_t bias_multipler_offset = conv_params->dest_offset - 1;
        while (bias_multipler_offset < inputs_len) {
            lea_buffer[bias_multipler_offset] = -0x8000; // _Q15(-1.0)
            bias_multipler_offset += conv_params->dest_offset;
        }
    }

    int16_t max_input_h = MIN_VAL(conv_params->input_h+tile_h-1, conv_params->input_h_last);
    int32_t max_h_end = int16_min(conv_params->input_h+tile_h+(layer_dims->kH-layer_dims->STRIDE_H), layer_dims->H)-1;
    int32_t h_start = int16_max(conv_params->input_h, 0);
#if ON_DEMAN_TILE_LOADING
    int32_t h_end = int16_min(h_start + layer_dims->kH, layer_dims->H) - 1;
#else
    int32_t h_end = max_h_end;
    load_ifm_tile_row(model, layer_dims, conv_params, h_start, h_end);
#endif
    for (int16_t cur_input_h = conv_params->input_h; cur_input_h <= max_input_h; cur_input_h += layer_dims->STRIDE_H) {
#if ON_DEMAN_TILE_LOADING
        load_ifm_tile_row(model, layer_dims, conv_params, h_start, h_end);
        h_start = h_end + 1;
        h_end = int16_min(h_end + layer_dims->STRIDE_H, max_h_end);
#endif

        my_printf_debug("Loaded inputs" NEWLINE);
        // state = 0 as state bits are already removed by my_offset_q15 above
        dump_matrix_debug(lea_buffer, inputs_len, ValueInfo(conv_params->conv_input, nullptr), false);

        // filter_idx is set to initial_c in handle_conv
        convTask(cur_input_h, layer_dims, conv_params);
        // reset here for further processing
        conv_params->filter_idx = conv_params->filter_tile_index * conv_params->flags->conv.output_tile_c;
    }
    return tile_h;
}

void alloc_conv(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node* node, NodeFlags* node_flags, const NodeFlags* orig_node_flags) {
    const ParameterInfo *conv_input = input[0], *conv_filter = input[1];

    /* input: N x C x H x W, filter: M x C x kH x kW */
    const uint16_t CHANNEL = conv_filter->dims[1], H = conv_input->dims[2], W = conv_input->dims[3];
    uint16_t OUTPUT_CHANNEL = conv_filter->dims[0];

    ConvTaskParams *conv_params = &conv_params_obj;
    ConvLayerDimensions* layer_dims = &(conv_params->layer_dims);

    conv_params->model = model;
    conv_params->flags = node_flags;
    conv_params->orig_flags = orig_node_flags;

    layer_dims->kH = conv_filter->dims[2];
    layer_dims->kW = conv_filter->dims[3];

    layer_dims->STRIDE_H = conv_params->flags->conv.strides[0];
    layer_dims->STRIDE_W = conv_params->flags->conv.strides[1];

    conv_params->pState_len = orig_node_flags->conv.pState_len;
    conv_params->group = conv_params->flags->conv.group;

    const uint8_t* pads = conv_params->flags->conv.pads;
    enum { PAD_H_BEGIN = 0, PAD_W_BEGIN = 1, PAD_H_END = 2, PAD_W_END = 3 };
    conv_params->input_h_first = -pads[PAD_H_BEGIN];
    conv_params->input_w_first = -pads[PAD_W_BEGIN];
    conv_params->input_h_last = H + pads[PAD_H_END] - layer_dims->kH;
    conv_params->input_w_last = W + pads[PAD_W_END] - layer_dims->kW;

    layer_dims->OUTPUT_H = (conv_params->input_h_last - conv_params->input_h_first) / layer_dims->STRIDE_H + 1;
    layer_dims->OUTPUT_W = (conv_params->input_w_last - conv_params->input_w_first) / layer_dims->STRIDE_W + 1;

#if !JAPARI
    // skip the check for JAPARI as it is too complex
    MY_ASSERT(conv_input->dims[1] == conv_filter->dims[1] * conv_params->group);
    MY_ASSERT(conv_params->group == 1 || conv_params->group == conv_input->dims[1]);
#endif

#if JAPARI
    start_cpu_counter(offsetof(Counters, stripping));
    conv_params->force_align_footprints = (OUTPUT_CHANNEL % BATCH_SIZE != 0);
    OUTPUT_CHANNEL = extend_for_footprints(OUTPUT_CHANNEL, conv_params->force_align_footprints);
    stop_cpu_counter();
#endif
    conv_params->n_tiles_c = (CHANNEL + conv_params->flags->conv.input_tile_c - 1) / conv_params->flags->conv.input_tile_c;
#if STATEFUL
    start_cpu_counter(offsetof(Counters, memory_layout));
    if (conv_params->flags->conv.output_tile_c % BATCH_SIZE) {
        conv_params->output_padding = BATCH_SIZE - conv_params->flags->conv.output_tile_c % BATCH_SIZE;
    } else {
        conv_params->output_padding = 0;
    }
    OUTPUT_CHANNEL += conv_params->output_padding;
    stop_cpu_counter();
#endif
    my_printf_debug("input_tile_c=%d, output_tile_c=%d" NEWLINE, conv_params->flags->conv.input_tile_c, conv_params->flags->conv.output_tile_c);

    /* XXX: extend flags; assume dilation=(1, 1) for now */
    output->slot = get_next_slot(model, conv_input);
    output->params_len = conv_params->n_tiles_c * layer_dims->OUTPUT_H * layer_dims->OUTPUT_W * OUTPUT_CHANNEL * sizeof(int16_t);
    output->dims[0] = 1;
    output->dims[1] = OUTPUT_CHANNEL;
    output->dims[2] = layer_dims->OUTPUT_H;
    output->dims[3] = layer_dims->OUTPUT_W;
    output->scale = conv_input->scale * conv_filter->scale;
    output->param_flags |= TRANSPOSED;
#if STATEFUL
    start_cpu_counter(offsetof(Counters, embedding));
    if (conv_input->slot == SLOT_TEST_SET) {
        output->scale.shift++;
    }
    stop_cpu_counter();
#endif
}

#if RuntimeConfiguration == Exhaustive
static void adapt_conv(ConvTaskParams* conv_params) {
    const uint16_t N_AVERAGE = 1;    // Using average of N_AVERAGE end-to-end inference latencies for each tile size
    uint16_t iteration_idx = conv_params->model->run_counter / N_AVERAGE;
    uint16_t N_ITERATIONS = EXHAUSTIVE_LOOKUP_TABLE_DATA_LEN / sizeof(ExhaustiveLookupTable);
    iteration_idx %= N_ITERATIONS;
    const ExhaustiveLookupTable* current_config = reinterpret_cast<const ExhaustiveLookupTable*>(exhaustive_lookup_table_data) + iteration_idx;
    uint16_t node_idx = conv_params->output->parameter_info_idx - N_INPUT;
    if (current_config->node_idx == node_idx) {
        conv_params->flags->conv.output_tile_c = current_config->conv_output_tile_c;
    } else {
        conv_params->flags->conv.output_tile_c = conv_params->orig_flags->conv.output_tile_c;
    }
    my_printf_debug("Selected output_tile_c: %d" NEWLINE, conv_params->flags->conv.output_tile_c);

    if (conv_params->model->run_counter % N_AVERAGE == 0) {
        // my_printf("%d %d" NEWLINE, node_idx, conv_params->flags->conv.output_tile_c);
        // Start of an end-to-end inference
        if (node_idx == 0) {
            // notify_exhaustive_new_value();
        }
        // Start adjusting a new layer
        if (conv_params->flags->conv.output_tile_c == 4) {
            // notify_exhaustive_new_layer();
        }
    }
}
#endif

void handle_conv(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node* node, NodeFlags*, const NodeFlags*) {
    const ParameterInfo *conv_input = input[0], *conv_filter = input[1], *conv_bias = (node->inputs_len == 3) ? input[2] : nullptr;
    my_printf_debug("Conv!" NEWLINE);

    /* input: N x C x H x W, filter: M x C x kH x kW */
    const uint16_t H = conv_input->dims[2], W = conv_input->dims[3],
                   CHANNEL = conv_filter->dims[1];

    ConvTaskParams *conv_params = &conv_params_obj;
    ConvLayerDimensions* layer_dims = &(conv_params->layer_dims);

    my_printf_debug("n_tiles_c = %d" NEWLINE, conv_params->n_tiles_c);

    conv_params->conv_input = conv_input;
    conv_params->conv_filter = conv_filter;
    conv_params->conv_bias = conv_bias;
    conv_params->output = output;
    conv_params->filter_buffer_addr = NULL;
    conv_params->cached_filter_idx = -1;
    layer_dims->H = H;
    layer_dims->W = W;
#if JAPARI
    start_cpu_counter(offsetof(Counters, stripping));
    conv_params->conv_input_has_footprints = has_footprints(conv_input);
    stop_cpu_counter();
#endif

    layer_dims->CHANNEL = CHANNEL;
    layer_dims->OUTPUT_CHANNEL = output->dims[1];
    layer_dims->N_FILTERS = conv_filter->dims[0];

    conv_params->input_tile_c_offset = 0;
    conv_params->input_tile_c_index = 0;
    conv_params->input_h = conv_params->input_h_first;
    conv_params->input_w = conv_params->input_w_first;
    conv_params->filter_tile_index = 0;
    conv_params->filter_idx = 0;
#if INTERMITTENT
    start_cpu_counter(offsetof(Counters, progress_seeking));
    uint32_t first_unfinished_job_idx = run_recovery(model, output);
    uint32_t first_unfinished_value_offset = batch_start(job_index_to_offset(output, first_unfinished_job_idx));
#if DYNBAL_REPORT_PARAMETERS
    conv_params->first_unfinished_job_idx = first_unfinished_job_idx;
    conv_params->reported = 0;
#endif

    fix_first_unfinished_value_offset(model, &first_unfinished_value_offset);

#if INDIRECT_RECOVERY
    start_cpu_counter(offsetof(Counters, state_query));
    find_initial_state_bit(&conv_params->old_output_offset, &conv_params->turning_point_idx, &conv_params->next_turning_point,
                           &conv_params->cur_slot_info, job_index_to_offset(output, first_unfinished_job_idx), model, output);
    stop_cpu_counter();

    my_printf_debug("old_output_offset = %d" NEWLINE, conv_params->old_output_offset);
#endif

    uint16_t cur_output_tile_c = conv_params->flags->conv.output_tile_c;
#if JAPARI
    start_cpu_counter(offsetof(Counters, embedding));
    cur_output_tile_c = extend_for_footprints(cur_output_tile_c, conv_params->force_align_footprints);
    stop_cpu_counter();
#endif
    uint16_t slice_size_input_channel_tiling = layer_dims->OUTPUT_W * layer_dims->OUTPUT_H * layer_dims->OUTPUT_CHANNEL;

    conv_params->input_tile_c_index = first_unfinished_value_offset / slice_size_input_channel_tiling;
    // Not extending for JAPARI footprints here as input_tile_c_offset will be extended later
    conv_params->input_tile_c_offset = conv_params->input_tile_c_index * conv_params->flags->conv.input_tile_c;
    first_unfinished_value_offset %= slice_size_input_channel_tiling;

    conv_params->filter_tile_index = (first_unfinished_value_offset % layer_dims->OUTPUT_CHANNEL) / cur_output_tile_c;
    conv_params->filter_idx = first_unfinished_value_offset % layer_dims->OUTPUT_CHANNEL;
#if JAPARI
    conv_params->filter_idx = conv_params->filter_idx / (BATCH_SIZE + 1) * BATCH_SIZE;
#endif

    first_unfinished_value_offset /= layer_dims->OUTPUT_CHANNEL;

    conv_params->input_w += first_unfinished_value_offset / layer_dims->OUTPUT_H * conv_params->layer_dims.STRIDE_W;
    first_unfinished_value_offset %= layer_dims->OUTPUT_H;

    conv_params->input_h += first_unfinished_value_offset * conv_params->layer_dims.STRIDE_H;

    my_printf_debug("initial output N = %d" NEWLINE, conv_params->input_tile_c_index);
    my_printf_debug("initial output H = %d" NEWLINE, (conv_params->input_h - conv_params->input_h_first) / conv_params->layer_dims.STRIDE_H);
    my_printf_debug("initial output W = %d" NEWLINE, (conv_params->input_w - conv_params->input_w_first) / conv_params->layer_dims.STRIDE_W);
    my_printf_debug("initial output C = %d" NEWLINE, conv_params->filter_idx);
    // = happens when all values are finished
    MY_ASSERT(conv_params->input_tile_c_index <= conv_params->n_tiles_c);
    stop_cpu_counter();
#endif

#if RuntimeConfiguration == Exhaustive
    if (first_unfinished_job_idx == 0) {
        // Starting a new layer
        adapt_conv(conv_params);
        commit_node_flags(conv_params->flags);
    }
#endif

#if INTERMITTENT && (RuntimeConfiguration == DynBal || ENABLE_DEMO_COUNTERS)
    update_progress_indicator_conv(node, conv_params->flags, conv_params->orig_flags, conv_params->layer_dims, first_unfinished_job_idx);
#endif

#if DYNBAL_REPORT_PARAMETERS
    if (first_unfinished_job_idx == 0) {
        uint16_t node_idx = conv_params->output->parameter_info_idx - N_INPUT;
        my_printf("%d,%d,%d,", node_idx, conv_params->flags->conv.input_tile_c, conv_params->flags->conv.output_tile_c);
    }
#endif

    conv_params->n_tiles_c = upper_gauss(CHANNEL, conv_params->flags->conv.input_tile_c);
    output->params_len = conv_params->n_tiles_c * layer_dims->OUTPUT_H * layer_dims->OUTPUT_W * layer_dims->OUTPUT_CHANNEL * sizeof(int16_t);

#if ENABLE_DEMO_COUNTERS
    if (first_unfinished_job_idx == 0) {
        uint16_t last_reported_layer_idx = static_cast<uint16_t>(-1);
        read_from_nvm(&last_reported_layer_idx, LAST_REPORTED_LAYER_IDX_OFFSET, sizeof(uint16_t));
        // really report costs only once per layer to avoid non-termination
        if (last_reported_layer_idx != conv_params->model->layer_idx) {
            InferenceStats* stats = load_inference_stats_from_nvm(InferenceStatsOpType::Conv);
            const UsageSpanConv usage_span(*layer_dims, conv_params->flags->conv.input_tile_c, conv_params->flags->conv.output_tile_c, stats->power_cycle_energy);

            uint32_t data_reuse_cost = usage_span.data_reuse_cost(UsageSpanConv::ParameterDimension::InputTileChannel, conv_params->flags->conv.input_tile_c);
            uint32_t data_refetch_cost = usage_span.data_refetch_cost(UsageSpanConv::ParameterDimension::InputTileChannel, conv_params->flags->conv.input_tile_c);
            my_printf("U,%" PRIu32 ",%" PRIu32 NEWLINE, data_reuse_cost/1024, data_refetch_cost/1024);

            last_reported_layer_idx = conv_params->model->layer_idx;
            write_to_nvm(&last_reported_layer_idx, LAST_REPORTED_LAYER_IDX_OFFSET, sizeof(uint16_t));
        }
    }
#endif

    int16_t input_channels = conv_filter->dims[1];
    for (; conv_params->input_tile_c_offset < input_channels; conv_params->input_tile_c_offset += conv_params->flags->conv.input_tile_c) {
        conv_params->cur_input_tile_c = MIN_VAL(conv_params->flags->conv.input_tile_c, input_channels - conv_params->input_tile_c_offset);
        conv_params->cur_filter_tile_c = conv_params->cur_input_tile_c;
#if JAPARI
        start_cpu_counter(offsetof(Counters, embedding));
        conv_params->input_tile_c_offset_with_footprints = extend_for_footprints(conv_params->input_tile_c_offset);
        stop_cpu_counter();
#endif
        my_printf_debug("cur_input_tile_c = %d" NEWLINE, conv_params->cur_input_tile_c);
        conv_params->dest_offset = layer_dims->kW * conv_params->cur_input_tile_c;
        if (conv_params->group == 1) {
            // +1 for bias
            conv_params->dest_offset++;
        }
        /* MSP430 LEA requires length to be even */
        conv_params->truncated = (conv_params->dest_offset / 2 * 2 != conv_params->dest_offset);
        if (conv_params->truncated && conv_params->group == 1) {
            // when CHANNEL * kH * kW is odd, CHANNEL * kW (dest_offset) is
            // also odd, so dummy values are needed between slices to make
            // addresses even.
            // a dummy value for each slice (kW * CHANNEL q15 values)
            conv_params->dest_offset++;
        }
        conv_params->filter_offset = layer_dims->kH * conv_params->dest_offset;

        while (true) {
            for (; conv_params->input_w <= conv_params->input_w_last; conv_params->input_w += conv_params->layer_dims.STRIDE_W) {
                for (; conv_params->input_h <= conv_params->input_h_last;) {
                    conv_params->input_h += handle_conv_inner_loop(model, layer_dims, conv_params);
                }
                conv_params->input_h = conv_params->input_h_first;
                report_progress();
            }
            conv_params->input_w = conv_params->input_w_first;
            conv_params->filter_tile_index++;
            if (conv_params->filter_tile_index * conv_params->flags->conv.output_tile_c >= layer_dims->N_FILTERS) {
                break;
            }
            conv_params->filter_idx = conv_params->filter_tile_index * conv_params->flags->conv.output_tile_c;
#if INDIRECT_RECOVERY
            start_cpu_counter(offsetof(Counters, state_query));
            uint32_t new_output_offset = conv_params->input_tile_c_index * layer_dims->OUTPUT_CHANNEL * layer_dims->OUTPUT_H * layer_dims->OUTPUT_W;
#if JAPARI
            start_cpu_counter(offsetof(Counters, embedding));
            new_output_offset += extend_for_footprints(conv_params->filter_idx);
            stop_cpu_counter();
#else
            new_output_offset += conv_params->filter_idx;
#endif
            find_initial_state_bit(&conv_params->old_output_offset, &conv_params->turning_point_idx, &conv_params->next_turning_point, &conv_params->cur_slot_info,
                                   new_output_offset, model, output);
            stop_cpu_counter();
#endif
        }
        conv_params->filter_idx = conv_params->filter_tile_index = 0;

        conv_params->input_tile_c_index++;
#if INDIRECT_RECOVERY
        start_cpu_counter(offsetof(Counters, state_query));
        find_initial_state_bit(&conv_params->old_output_offset, &conv_params->turning_point_idx, &conv_params->next_turning_point, &conv_params->cur_slot_info,
                               conv_params->input_tile_c_index * layer_dims->OUTPUT_CHANNEL * layer_dims->OUTPUT_H * layer_dims->OUTPUT_W, model, output);
        stop_cpu_counter();
#endif
    }

#if INDIRECT_RECOVERY
    start_cpu_counter(offsetof(Counters, table_updates));
    flip_state_bit(model, output);
    stop_cpu_counter();
#endif

    my_printf_debug("handle_conv output" NEWLINE);
    dump_params_nhwc_debug(model, output, node->output_name);
}

void alloc_convmerge(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node*, NodeFlags*, const NodeFlags*) {
    const ParameterInfo *data = input[0];

    uint16_t OUTPUT_CHANNEL = data->dims[1],
             OUTPUT_H = data->dims[2],
             OUTPUT_W = data->dims[3];

    output->slot = get_next_slot(model, data);
    output->params_len = OUTPUT_CHANNEL * OUTPUT_H * OUTPUT_W * sizeof(int16_t);
    output->param_flags &= (~TRANSPOSED);
}

#if STATEFUL
struct ConvMergeInputChunkHandlerParams {
    int16_t *to_add;
    uint16_t input_offset;
};

void ConvMergeInputChunkHandler(uint32_t range_offset, uint16_t range_len, int8_t state_bit, void* _params) {
    ConvMergeInputChunkHandlerParams* params = reinterpret_cast<ConvMergeInputChunkHandlerParams*>(_params);
    my_printf_debug("input range_offset=%d range_len=%d state_bit=%d" NEWLINE, range_offset, range_len, state_bit);
    int16_t *to_offset = params->to_add + range_offset - params->input_offset;
    my_offset_q15_batched(to_offset, -state_bit*0x4000, to_offset, range_len);
}
#endif

#if JAPARI
struct ConvMergeOutputChunkHandlerParams {
    uint32_t tiling_results_offset;
};

void ConvMergeOutputChunkHandler(uint32_t range_offset, uint16_t range_len, int8_t state_bit, void* _params) {
    ConvMergeOutputChunkHandlerParams* params = reinterpret_cast<ConvMergeOutputChunkHandlerParams*>(_params);
    my_printf_debug("output range_offset=%d range_len=%d state_bit=%d" NEWLINE, range_offset, range_len, state_bit);
    int16_t *to_offset = lea_buffer + range_offset - params->tiling_results_offset;
    uint16_t n_footprints = (range_len + BATCH_SIZE) / (BATCH_SIZE + 1);
    int16_t* footprint_buffer = lea_buffer + (LEA_BUFFER_SIZE - n_footprints) / 2 * 2;
    my_fill_q15(-state_bit, footprint_buffer, n_footprints);
    my_interleave_q15(footprint_buffer, BATCH_SIZE - (range_offset % (BATCH_SIZE + 1)), BATCH_SIZE + 1, to_offset, n_footprints);
}
#endif

void handle_convmerge(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node* node, NodeFlags*, const NodeFlags*) {
    // Do not use conv_params here as its intialization in alloc_conv and
    // handle_conv might be skipped if the Conv node has finished.
    const ParameterInfo *data = input[0];
    uint16_t OUTPUT_CHANNEL = data->dims[1],
             OUTPUT_H = data->dims[2],
             OUTPUT_W = data->dims[3];

    my_printf_debug("ConvMerge!" NEWLINE);

    uint8_t n_tiles_c = data->params_len / sizeof(int16_t) / (OUTPUT_CHANNEL * OUTPUT_H * OUTPUT_W);

    MY_ASSERT(n_tiles_c);

    uint32_t tiling_results_len = OUTPUT_CHANNEL * OUTPUT_H * OUTPUT_W;

    uint16_t chunk_len = OUTPUT_CHANNEL;
    uint16_t output_h = 0, output_w = 0, chunk_offset = 0;
#if INTERMITTENT
    start_cpu_counter(offsetof(Counters, progress_seeking));
    uint32_t first_unfinished_job_idx = run_recovery(model, output);
    uint32_t first_unfinished_value_offset = batch_start(job_index_to_offset(output, first_unfinished_job_idx));

    MY_ASSERT(chunk_len * n_tiles_c < LEA_BUFFER_SIZE);

    // value offset = output_h * OUTPUT_W * chunk_len + output_w * chunk_len + chunk_offset;
    chunk_offset = first_unfinished_value_offset % chunk_len;
    first_unfinished_value_offset /= chunk_len;
    output_w = first_unfinished_value_offset % OUTPUT_W;
    first_unfinished_value_offset /= OUTPUT_W;
    output_h = first_unfinished_value_offset;
    stop_cpu_counter();
#endif

    // Here IFM and OFM have different data layouts as I do the conversion in this handler
    uint32_t input_offset = output_w * OUTPUT_H * OUTPUT_CHANNEL +
                            output_h * OUTPUT_CHANNEL +
                            chunk_offset; // NWHC
    uint32_t output_offset = output_h * OUTPUT_W * OUTPUT_CHANNEL +
                             output_w * OUTPUT_CHANNEL +
                             chunk_offset; // NHWC

#if INDIRECT_RECOVERY
    start_cpu_counter(offsetof(Counters, state_query));
    int16_t old_embedding_offset;
    uint8_t output_turning_point_idx;
    uint16_t next_output_turning_point;
    SlotInfo *cur_output_slot_info;

    find_initial_state_bit(&old_embedding_offset, &output_turning_point_idx, &next_output_turning_point,
                           &cur_output_slot_info, output_offset, model, output);

    my_printf_debug("old_embedding_offset = %d" NEWLINE, old_embedding_offset);
    stop_cpu_counter();
#endif

    for (; output_h < OUTPUT_H;) {
        for (; output_w < OUTPUT_W; output_w++) {
            uint16_t real_chunk_len = chunk_len - chunk_offset;
            my_printf_debug("real_chunk_len = %d" NEWLINE, real_chunk_len);
            for (uint16_t input_tile_c_index = 0; input_tile_c_index < n_tiles_c; input_tile_c_index++) {
                int16_t *to_add = lea_buffer + input_tile_c_index * chunk_len;
                uint16_t cur_input_offset = input_tile_c_index * tiling_results_len + input_offset;
                my_memcpy_from_param(model, to_add, data, cur_input_offset, real_chunk_len * sizeof(int16_t));
#if JAPARI && ENABLE_COUNTERS
                add_counter(offsetof(Counters, data_loading), (real_chunk_len/2)*(4*8));
#endif
#if STATEFUL
                start_cpu_counter(offsetof(Counters, stripping));
                ConvMergeInputChunkHandlerParams params({to_add, cur_input_offset});
                iterate_chunks(model, data, cur_input_offset, real_chunk_len, ConvMergeInputChunkHandler, &params);
                stop_cpu_counter();
#endif
                my_printf_debug(NEWLINE "Input offset %d, input tile %d, output offset %d" NEWLINE, cur_input_offset, input_tile_c_index, output_offset);
                my_printf_debug("Added chunk" NEWLINE);
                dump_matrix_debug(to_add, real_chunk_len, ValueInfo(data));
                if (input_tile_c_index != 0) {
                    my_add_q15(lea_buffer, to_add, lea_buffer, real_chunk_len);
                }
            }
#if INDIRECT_RECOVERY

#if STATEFUL
            start_cpu_counter(offsetof(Counters, state_query));
            fill_state_offsets(output_offset, real_chunk_len, &old_embedding_offset, &output_turning_point_idx, &next_output_turning_point, cur_output_slot_info);
            stop_cpu_counter();

            start_cpu_counter(offsetof(Counters, embedding));
            update_states(lea_buffer, real_chunk_len, true);
            stop_cpu_counter();
#elif JAPARI
            start_cpu_counter(offsetof(Counters, embedding));
            ConvMergeOutputChunkHandlerParams params({output_offset});
            iterate_chunks(model, output, output_offset, real_chunk_len, ConvMergeOutputChunkHandler, &params);
            stop_cpu_counter();
#endif

            my_printf_debug("After writing state bits in [%d, %d)" NEWLINE, output_offset, output_offset + real_chunk_len);
            dump_matrix_debug(lea_buffer, real_chunk_len, ValueInfo(output));
#endif

            my_memcpy_to_param(output, output_offset, lea_buffer, real_chunk_len * sizeof(int16_t), 0, true);
#if HAWAII
            hawaii_record_footprints(model, real_chunk_len);
#endif
            output_offset += real_chunk_len;
            input_offset += OUTPUT_H * OUTPUT_CHANNEL - chunk_offset; // NWHC
            chunk_offset = 0;
        }
        output_w = 0;
        output_h++;
        input_offset = output_h * OUTPUT_CHANNEL; // NWHC, where only output_h is nonzero at this point

        report_progress();
    }

    my_printf_debug("After merging tiling results" NEWLINE);

#if INDIRECT_RECOVERY
    start_cpu_counter(offsetof(Counters, table_updates));
    flip_state_bit(model, output);
    stop_cpu_counter();
#endif

    dump_params_nhwc_debug(model, output, node->output_name);
}
