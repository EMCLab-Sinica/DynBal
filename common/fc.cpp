#include <cstddef>
#include <cstdint>
#include "cnn_common.h"
#include "counters.h"
#include "data.h"
#include "dynbal-fc.h"
#include "fc.h"
#include "platform.h"
#include "my_debug.h"
#include "op_utils.h"
#include "my_dsplib.h"
#include "intermittent-cnn.h"

/**
 * For fully-connected layers, which are implemented via Gemm in ONNX.
 */

void alloc_gemm(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node* node, NodeFlags* node_flags, const NodeFlags*) {
    const ParameterInfo *A = input[0], *B = input[1];

    MY_ASSERT(A->dims[0] == 1);

    output->dims[0] = A->dims[0];
#if JAPARI
    start_cpu_counter(offsetof(Counters, embedding));
    output->dims[1] = B->dims[1] / BATCH_SIZE * (BATCH_SIZE + 1) + B->dims[1] % BATCH_SIZE;
    stop_cpu_counter();
#else
    output->dims[1] = B->dims[1];
#endif
    output->slot = get_next_slot(model, A);
    output->scale = A->scale * B->scale;

    uint16_t output_len = output->dims[0] * output->dims[1];

    output->params_len = output_len * upper_gauss(B->dims[0], node_flags->gemm.tile_channel) * sizeof(int16_t);
}

int16_t* const weights_tmp = op_buffer;

void handle_gemm(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node* node, NodeFlags* node_flags, const NodeFlags* orig_node_flags) {
    const ParameterInfo *A = input[0], *B = input[1], *matC = input[2];

    my_printf_debug("Gemm! A: (%dx%d), B: (%dx%d)" NEWLINE,
              A->dims[0], A->dims[1], B->dims[0], B->dims[1]);

    int16_t A_len = A->dims[0] * A->dims[1] + 2,
            output_len = output->dims[0] * output->dims[1];

    int16_t *buffer_a = lea_buffer,
            *buffer_temp = buffer_a + (A_len + 1) / 2 * 2; // guarantee even addresses, making check_buffer_address happy
#if JAPARI
    start_cpu_counter(offsetof(Counters, embedding));
    buffer_temp += 2;
    int16_t* buffer_b = buffer_temp + extend_for_footprints(node_flags->gemm.tile_width);
    stop_cpu_counter();
#else
    int16_t* buffer_b = buffer_temp + node_flags->gemm.tile_width;
#endif
    make_buffer_aligned(&buffer_b);

    uint16_t i = 0, tile = 0, j = 0, j_with_footprints = 0;

#if INTERMITTENT
    start_cpu_counter(offsetof(Counters, progress_seeking));
    uint32_t first_unfinished_value_offset = job_index_to_offset(output, run_recovery(model, output));

#if INDIRECT_RECOVERY
    start_cpu_counter(offsetof(Counters, state_query));
    int16_t offset;
    uint16_t next_output_turning_point;
    uint8_t output_turning_point_idx;
    SlotInfo *output_slot_info;
    find_initial_state_bit(&offset, &output_turning_point_idx, &next_output_turning_point, &output_slot_info, first_unfinished_value_offset, model, output);
    stop_cpu_counter();
#endif

    first_unfinished_value_offset = batch_start(first_unfinished_value_offset);

    fix_first_unfinished_value_offset(model, &first_unfinished_value_offset);

    tile = first_unfinished_value_offset / output_len;
    i = tile * node_flags->gemm.tile_channel;
    j_with_footprints = first_unfinished_value_offset % output_len;

#if JAPARI
    start_cpu_counter(offsetof(Counters, embedding));
    j = j_with_footprints / (BATCH_SIZE + 1) * BATCH_SIZE;
    stop_cpu_counter();
#else
    j = j_with_footprints;
#endif

    stop_cpu_counter();

#if RuntimeConfiguration == DynBal
    FcLayerDimensions layer_dims;
    layer_dims.A_rows = A->dims[0];
    layer_dims.A_cols = A->dims[1];
    layer_dims.B_cols = B->dims[1];
    update_progress_indicator_fc(node, node_flags, orig_node_flags, layer_dims, first_unfinished_value_offset);

#if DYNBAL_REPORT_PARAMETERS
    if (first_unfinished_value_offset == 0) {
        uint16_t node_idx = output->parameter_info_idx - N_INPUT;
        my_printf("%d,%d,%d" NEWLINE, node_idx, node_flags->gemm.tile_channel, node_flags->gemm.tile_width);
    }
#endif

#endif

    my_printf_debug("tile_channel=%d, tile_width=%d" NEWLINE, node_flags->gemm.tile_channel, node_flags->gemm.tile_width);
    output->params_len = output_len * upper_gauss(B->dims[0], node_flags->gemm.tile_channel) * sizeof(int16_t);
    MY_ASSERT(node_flags->gemm.tile_width / BATCH_SIZE * BATCH_SIZE == node_flags->gemm.tile_width);

#endif

    for (; i < B->dims[0]; i += node_flags->gemm.tile_channel, tile++) {
        const uint16_t tile_channels = MIN_VAL(node_flags->gemm.tile_channel, B->dims[0] - i);
        const uint16_t extended_tile_channels = tile_channels + 2;

#if JAPARI
        start_cpu_counter(offsetof(Counters, stripping));
        bool need_skipping = has_footprints(A);
        if (need_skipping) {
            // somehow loading many pieces is faster than loading a chunk and moving values around to remove footprints, even with external FRAM
            uint16_t input_offset = extend_for_footprints(i);
            for (uint16_t idx = 0, output_idx = 0; output_idx < tile_channels; idx += BATCH_SIZE + 1, output_idx += BATCH_SIZE) {
                my_memcpy_from_param(model, buffer_a + output_idx, A, input_offset + idx, BATCH_SIZE * sizeof(uint16_t));
            }
        }
        stop_cpu_counter();
        if (!need_skipping)
#endif
        {
            my_memcpy_from_param(model, buffer_a, A, i, tile_channels * sizeof(uint16_t));
        }

#if STATEFUL
        start_cpu_counter(offsetof(Counters, stripping));
        if (A->slot != SLOT_TEST_SET) {
            for (int16_t *input_ptr = buffer_a + BATCH_SIZE - 1; input_ptr < buffer_a + tile_channels; input_ptr += BATCH_SIZE) {
                strip_state(input_ptr);
            }
        }
        stop_cpu_counter();
#endif
        buffer_a[tile_channels] = -0x8000;
        buffer_a[tile_channels + 1] = 0;

        my_printf_debug("Tile for A" NEWLINE);
        dump_matrix_debug(buffer_a, 1, extended_tile_channels, ValueInfo(A, model));

        int16_t output_offset = tile * output_len + j_with_footprints;

        for (; j < B->dims[1]; j += node_flags->gemm.tile_width) {
            int16_t tile_width = MIN_VAL(node_flags->gemm.tile_width, B->dims[1] - j);
            int16_t values_to_preserve = tile_width,
                    full_tile_width = tile_width;
#if JAPARI
            start_cpu_counter(offsetof(Counters, embedding));
            values_to_preserve = extend_for_footprints(tile_width);
            full_tile_width = (values_to_preserve + 1) / 2 * 2;
            stop_cpu_counter();
#endif
            int16_t *filter_ptr = buffer_b;
            my_fill_q15(0, filter_ptr, extended_tile_channels * full_tile_width);
            for (uint16_t row = 0; row < tile_width; row++) {
                MY_ASSERT(tile_channels <= OP_BUFFER_LEN);
                my_memcpy_from_param(model, weights_tmp,
                          B, (j + row) * B->dims[0] + i,
                          tile_channels * sizeof(uint16_t));
#if JAPARI
                my_interleave_q15(weights_tmp, extend_for_footprints(row), full_tile_width, filter_ptr, tile_channels);
#else
                my_interleave_q15(weights_tmp, row, full_tile_width, filter_ptr, tile_channels);
#endif
            }
            filter_ptr += tile_channels * full_tile_width;
#if JAPARI
            start_cpu_counter(offsetof(Counters, embedding));
            my_fill_q15(0, filter_ptr, 2 * full_tile_width);
            uint8_t processed_biases = 0, bias_offset = 0;
            for (uint16_t idx = 0; idx < values_to_preserve; idx++) {
                if (processed_biases == BATCH_SIZE) {
                    processed_biases = 0;
                    filter_ptr[idx] = param_state_bit(model, output, output_offset + idx);
                } else {
                    if (tile == 0) {
                        filter_ptr[idx] = -static_cast<int32_t>(get_q15_param(model, matC, bias_offset + j)) / A->scale.toFloat();
                    }
                    bias_offset++;
                    processed_biases++;
                }
            }
            stop_cpu_counter();
#else
            if (tile == 0) {
                for (uint16_t idx = 0; idx < values_to_preserve; idx++) {
                    filter_ptr[idx] = -static_cast<int32_t>(get_q15_param(model, matC, idx + j)) / A->scale.toFloat();
                }
            }
#endif

#if INDIRECT_RECOVERY
            start_cpu_counter(offsetof(Counters, state_query));
            fill_state_offsets(output_offset, tile_width, &offset, &output_turning_point_idx, &next_output_turning_point, output_slot_info);
            stop_cpu_counter();
#endif

#if STATEFUL
            start_cpu_counter(offsetof(Counters, embedding));
            update_states(filter_ptr, tile_width, false, true);
            stop_cpu_counter();
#endif

            my_printf_debug("Tile for B" NEWLINE);
            dump_matrix_debug(buffer_b, extended_tile_channels, full_tile_width, ValueInfo(B, model));
            my_matrix_mpy_q15(1, extended_tile_channels, extended_tile_channels, full_tile_width, buffer_a, buffer_b, buffer_temp,
                              output, output_offset, values_to_preserve, orig_node_flags->gemm.pState_len);
            my_printf_debug("matrix_mpy_results" NEWLINE);
            dump_matrix_debug(buffer_temp, full_tile_width, ValueInfo(output, model));
            my_printf_debug(NEWLINE);

            compare_vm_nvm(buffer_temp, model, output, output_offset, values_to_preserve);

            my_printf_debug("output_offset=%d" NEWLINE, output_offset);
#if HAWAII
            hawaii_record_footprints(model, values_to_preserve);
#endif
            output_offset += values_to_preserve;
        }
        j = j_with_footprints = 0;
    }

#if INDIRECT_RECOVERY
    start_cpu_counter(offsetof(Counters, table_updates));
    flip_state_bit(model, output);
    stop_cpu_counter();
#endif

    my_printf_debug("handle_gemm output" NEWLINE);
    dump_params_debug(model, output, node->output_name);
}

void alloc_gemmmerge(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node*, NodeFlags*, const NodeFlags*) {
    output->slot = get_next_slot(model, input[0]);
    int16_t output_len = output->dims[0] * output->dims[1];
    output->params_len = output_len * sizeof(int16_t);
}

void handle_gemmmerge(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node* node, NodeFlags* node_flags, const NodeFlags*) {
    const ParameterInfo *X = input[0];

    my_printf_debug("GemmMerge!" NEWLINE);

    int16_t output_len = X->dims[0] * X->dims[1];

    int16_t output_tile_size = node_flags->gemmmerge.tile_length;
    if (!output_tile_size) {
        output_tile_size = output_len;
    }
#if JAPARI
    start_cpu_counter(offsetof(Counters, embedding));
    output_tile_size = extend_for_footprints(output_tile_size);
    stop_cpu_counter();
#endif

    uint16_t merge_offset = 0;
#if INTERMITTENT
    start_cpu_counter(offsetof(Counters, progress_seeking));
    merge_offset = batch_start(job_index_to_offset(output, run_recovery(model, output)));
    stop_cpu_counter();
#endif

    int16_t *buffer_temp = lea_buffer,
            *buffer_gemm = buffer_temp + output_tile_size;
    make_buffer_aligned(&buffer_gemm);

    int16_t n_tiles = X->params_len / output_len / sizeof(int16_t);
    my_printf_debug("n_tiles=%d" NEWLINE, n_tiles);
    MY_ASSERT(n_tiles);

    for (; merge_offset < output_len; merge_offset += output_tile_size) {
        int16_t cur_tile_size = MIN_VAL(output_tile_size, output_len - merge_offset);
        my_fill_q15(0, buffer_gemm, cur_tile_size);

        for (uint16_t tile = 0; tile < n_tiles; tile++) {
            my_memcpy_from_param(model, buffer_temp, input[0], tile * output_len + merge_offset, cur_tile_size * sizeof(int16_t));
#if STATEFUL
            start_cpu_counter(offsetof(Counters, stripping));
            for (uint16_t idx = BATCH_SIZE - 1; idx < cur_tile_size; idx += BATCH_SIZE) {
                strip_state(buffer_temp + idx);
            }
            stop_cpu_counter();
#endif
            my_add_q15(buffer_gemm, buffer_temp, buffer_gemm, cur_tile_size);
            my_printf_debug("accumulated buffer_gemm" NEWLINE);
            dump_matrix_debug(buffer_gemm, cur_tile_size, ValueInfo(output, model));
        }

#if INDIRECT_RECOVERY
        start_cpu_counter(offsetof(Counters, embedding));
        OutputChunkHandlerParams params;
        params.buffer = buffer_gemm;
        params.buffer_offset = merge_offset;
        iterate_chunks(model, output, merge_offset, cur_tile_size, OutputChunkHandler, &params);
        stop_cpu_counter();
#endif
        my_printf_debug("buffer_gemm after adjusting states; merge_offset=%d" NEWLINE, merge_offset);
        dump_matrix_debug(buffer_gemm, cur_tile_size, ValueInfo(output, model));

        my_memcpy_to_param(output, merge_offset, buffer_gemm, cur_tile_size * sizeof(int16_t), 0, true);
#if HAWAII
        hawaii_record_footprints(model, cur_tile_size);
#endif
    }

#if INDIRECT_RECOVERY
    start_cpu_counter(offsetof(Counters, table_updates));
    flip_state_bit(model, output);
    stop_cpu_counter();
#endif

    my_printf_debug("handle_gemmmerge output" NEWLINE);
    dump_params_debug(model, output, node->output_name);
}
