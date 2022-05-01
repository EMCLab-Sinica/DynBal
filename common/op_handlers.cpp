#include <cstddef>
#include <cstdint>
#include "cnn_common.h"
#include "counters.h"
#include "data.h"
#include "op_utils.h"
#include "my_debug.h"
#include "intermittent-cnn.h"
#include "my_dsplib.h"
#include "platform.h"

#define RESHAPE_AUTO_DIM static_cast<uint16_t>(-1)

const uint8_t RELU_TILE_SIZE = 16;
static_assert(RELU_TILE_SIZE % BATCH_SIZE == 0, "Incorrect tile size for ReLU");

void alloc_relu(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node*) {
    const ParameterInfo *data = input[0];
    output->slot = get_next_slot(model, data);
}

void handle_relu(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node* node) {
    my_printf_debug("ReLu!" NEWLINE);

    const ParameterInfo *X = input[0];

    int16_t data_len = X->params_len / 2;

    uint16_t output_offset = 0;
#if INTERMITTENT
    start_cpu_counter(offsetof(Counters, progress_seeking));
    uint32_t first_unfinished_value_offset = batch_start(job_index_to_offset(output, run_recovery(model, output)));
    output_offset += first_unfinished_value_offset;

#if INDIRECT_RECOVERY
    start_cpu_counter(offsetof(Counters, state_query));
    uint16_t next_output_turning_point;
    int16_t offset;
    uint8_t output_turning_point_idx;
    SlotInfo *output_slot_info;
    find_initial_state_bit(&offset, &output_turning_point_idx, &next_output_turning_point, &output_slot_info,
                           first_unfinished_value_offset, model, output);
    offset = -offset;
    stop_cpu_counter();
#endif
    stop_cpu_counter();
#endif

    int16_t vals[32];
    uint16_t i = output_offset;
#if JAPARI
    start_cpu_counter(offsetof(Counters, embedding));
    const uint8_t real_relu_tile_size = extend_for_footprints(RELU_TILE_SIZE);
    stop_cpu_counter();
#else
    const uint8_t real_relu_tile_size = RELU_TILE_SIZE;
#endif
    for (; i < data_len; i += real_relu_tile_size) {
        uint8_t cur_tile_size = MIN_VAL(real_relu_tile_size, data_len - i);
        my_memcpy_from_param(model, vals, X, output_offset, cur_tile_size*sizeof(int16_t));
#if JAPARI && ENABLE_COUNTERS
        counters()->data_loading += (cur_tile_size/2)*(4*8);
#endif

#if STATEFUL
        start_cpu_counter(offsetof(Counters, stripping));
        for (uint8_t j = 0; j < cur_tile_size; j++) {
            if (offset_has_state(output_offset+j)) {
                strip_state(&vals[j]);
            }
            vals[j] *= 2;
        }
        stop_cpu_counter();
#endif

        for (uint8_t j = 0; j < cur_tile_size; j++) {
            vals[j] = MAX_VAL(vals[j], 0);
        }

#if INDIRECT_RECOVERY
        start_cpu_counter(offsetof(Counters, embedding));
#if STATEFUL
        const uint8_t embedding_shift = BATCH_SIZE;
#else
        const uint8_t embedding_shift = BATCH_SIZE + 1;
#endif
        for (uint8_t j = 0; j < cur_tile_size; j += embedding_shift) {
            uint8_t tile_last = j + embedding_shift - 1;
            start_cpu_counter(offsetof(Counters, state_query));
            check_next_turning_point(offset, output_turning_point_idx, next_output_turning_point, output_slot_info, output_offset + tile_last);
            stop_cpu_counter();
#if STATEFUL
            start_cpu_counter(offsetof(Counters, embedding));
            for (uint8_t k = j; k < tile_last; k++) {
                vals[k] /= 2;
            }
            vals[tile_last] = vals[tile_last] / 2 + offset;
            stop_cpu_counter();
#else
            vals[tile_last] = (offset > 0 ? 1 : -1);
#endif
        }
        stop_cpu_counter();
#endif

#if MY_DEBUG >= MY_DEBUG_VERBOSE
        my_printf_debug("output_offset=[% 6d, % 6d), output_val=", output_offset, output_offset+cur_tile_size);
        for (uint8_t j = 0; j < cur_tile_size; j++) {
            my_printf_debug("% 6d", vals[j]);
            if (j != cur_tile_size - 1) {
                my_printf_debug(", ");
            }
        }
        my_printf_debug(NEWLINE);
#endif

        my_memcpy_to_param(output, output_offset, vals, cur_tile_size*sizeof(int16_t), 0);
        output_offset += cur_tile_size;
#if HAWAII
        for (int8_t to_record = cur_tile_size; to_record > 0; to_record -= BATCH_SIZE) {
            write_hawaii_layer_footprint(model->layer_idx, BATCH_SIZE);
        }
#endif
    }

#if INDIRECT_RECOVERY
    start_cpu_counter(offsetof(Counters, table_updates));
    flip_state_bit(model, output);
    stop_cpu_counter();
#endif

    my_printf_debug("handle_relu output" NEWLINE);
    dump_params_nhwc_debug(model, output, node->output_name);
}

void handle_reshape(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node*) {
    my_printf_debug("Reshape!" NEWLINE);

    const ParameterInfo *data = input[0], *shape = input[1];
    /*
     * At most one dimension of the new shape can be -1. In this case, the
     * value is inferred from the size of the tensor and the remaining
     * dimensions.
     *
     * A dimension could also be 0, in which case the actual dimension value
     * is unchanged (i.e. taken from the input tensor).
     * */
    for (uint8_t i = 0; i < 4 && i < shape->dims[0]; i++) {
        output->dims[i] = get_int64_param(shape, i);
        if (!output->dims[i]) {
            output->dims[i] = data->dims[i];
        }
    }
    for (uint8_t i = shape->dims[0]; i < 4; i++) {
        output->dims[i] = 0;
    }
    uint16_t inferred_dim = output->params_len / sizeof(int16_t);
    int8_t auto_idx = -1;
#if JAPARI
    start_cpu_counter(offsetof(Counters, embedding));
    uint8_t last_dim_idx;
    for (uint8_t i = 0; i < 4; i++) {
        if (output->dims[i]) {
            last_dim_idx = i;
        }
    }
    stop_cpu_counter();
#endif
    for (uint8_t i = 0; i < 4; i++) {
        if (output->dims[i] != RESHAPE_AUTO_DIM && output->dims[i] != 0) {
#if JAPARI
            if (i == last_dim_idx && data->slot != SLOT_TEST_SET) {
                inferred_dim /= extend_for_footprints(output->dims[i]);
            } else
#endif
            {
                inferred_dim /= output->dims[i];
            }
        } else if (output->dims[i] == RESHAPE_AUTO_DIM) {
            auto_idx = i;
        }
    }
    if (auto_idx != -1) {
        output->dims[auto_idx] = inferred_dim;
    }
}

void handle_squeeze(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node* node) {
    my_printf_debug("Squeeze!" NEWLINE);

    uint8_t axes = node->flags.squeeze.axes;
    // If axes is not provided, all the single dimensions will be removed from the shape.
    // https://github.com/onnx/onnx/blob/master/docs/Operators.md#squeeze
    uint8_t j = 0;
    if (axes == 0) {
        for (uint8_t i = 0; i < 4; i++) {
            if (input[0]->dims[i] != 1) {
                output->dims[j] = input[0]->dims[i];
                j++;
            }
        }
    } else {
        for (uint8_t i = 0; i < 4; i++) {
            if (axes & (1 << i)) {
#if !JAPARI
                MY_ASSERT(input[0]->dims[i] == 1);
#endif
            } else {
                output->dims[j] = input[0]->dims[i];
                j++;
            }
        }
    }
    for (; j < 4; j++) {
        output->dims[j] = 0;
    }
}

void handle_unsqueeze(Model* model, const ParameterInfo* input[], ParameterInfo* output, const Node* node) {
    my_printf_debug("Unsqueeze!" NEWLINE);
    uint8_t axes = node->flags.squeeze.axes;
    uint8_t input_dim_offset = 0, output_dim_offset = 0;
    for (uint8_t i = 0; i < 4; i++) {
        if (axes & (1 << i)) {
            output->dims[output_dim_offset] = 1;
            output_dim_offset++;
        } else {
            output->dims[output_dim_offset] = input[0]->dims[input_dim_offset];
            input_dim_offset++;
            output_dim_offset++;
        }
    }
}

void alloc_concat(Model* model, const ParameterInfo *input[], ParameterInfo* output, const Node* node) {
    // Only channel concatenation is supported for now
    MY_ASSERT(node->flags.concat.axis == 1);

    output->dims[1] = 0;
    for (uint8_t input_idx = 0; input_idx < node->inputs_len; input_idx++) {
        const ParameterInfo* inp = input[input_idx];
        MY_ASSERT(inp->dims[1] <= LEA_BUFFER_SIZE);
#if JAPARI
        // Only support simple cases for now
        MY_ASSERT(inp->dims[1] % (BATCH_SIZE + 1) == 0);
#elif STATEFUL
        MY_ASSERT(inp->dims[1] % BATCH_SIZE == 0);
#endif
        output->dims[1] += inp->dims[1];
        output->scale = (inp->scale > output->scale) ? inp->scale : output->scale;
    }

    output->params_len = sizeof(int16_t);
    for (uint8_t dim_idx = 0; (dim_idx < 4) && output->dims[dim_idx]; dim_idx++) {
        output->params_len *= output->dims[dim_idx];
    }

    output->slot = get_next_slot(model, input[0]);
}

void handle_concat(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node* node) {
    my_printf_debug("Concat!" NEWLINE);

    uint32_t output_offset;
    uint16_t hw = 0;
#if INTERMITTENT
    start_cpu_counter(offsetof(Counters, progress_seeking));
    uint32_t first_unfinished_job_idx = run_recovery(model, output);
    output_offset = batch_start(job_index_to_offset(output, first_unfinished_job_idx));
    hw = output_offset / output->dims[1];

#if INDIRECT_RECOVERY
    start_cpu_counter(offsetof(Counters, state_query));
    uint16_t next_output_turning_point;
    int16_t offset;
    uint8_t output_turning_point_idx;
    SlotInfo *output_slot_info;
    find_initial_state_bit(&offset, &output_turning_point_idx, &next_output_turning_point, &output_slot_info,
                           output_offset, model, output);
    stop_cpu_counter();
#endif

#endif

    for (; hw < output->dims[2] * output->dims[3]; hw++) {
        uint16_t already_copied = output_offset - hw * output->dims[1];
        for (uint8_t input_idx = 0; input_idx < node->inputs_len; input_idx++) {
            const ParameterInfo* inp = input[input_idx];
            const int16_t input_channels = inp->dims[1];
            if (already_copied >= input_channels) {
                already_copied -= input_channels;
                continue;
            }
            uint16_t to_copy = input_channels - already_copied;
#if INDIRECT_RECOVERY
            start_cpu_counter(offsetof(Counters, state_query));
            check_next_turning_point(offset, output_turning_point_idx, next_output_turning_point, output_slot_info, output_offset);
            stop_cpu_counter();
#endif
            my_memcpy_from_param(model, lea_buffer, inp, hw * input_channels + already_copied, to_copy * sizeof(int16_t));
#if STATEFUL
            for (uint16_t idx = BATCH_SIZE - 1; idx < to_copy; idx += BATCH_SIZE) {
                strip_state(lea_buffer + idx);
            }
#endif
            if (inp->scale != output->scale) {
                int16_t scaleFract;
                uint8_t shift;
                float_to_scale_params(&scaleFract, &shift, output->scale/inp->scale);
                my_scale_q15(lea_buffer, scaleFract, shift, lea_buffer, to_copy * sizeof(int16_t));
            }

            start_cpu_counter(offsetof(Counters, embedding));
            update_states(lea_buffer, to_copy, output_offset, offset, next_output_turning_point, true);
            stop_cpu_counter();

            my_memcpy_to_param(output, output_offset, lea_buffer, to_copy * sizeof(int16_t), 0);
            output_offset += to_copy;
            my_printf_debug("Copied %u values" NEWLINE, to_copy);
#if HAWAII
            write_hawaii_layer_footprint(model->layer_idx, to_copy);
#endif
        }
    }

    dump_params_nhwc_debug(model, output);

#if INDIRECT_RECOVERY
    start_cpu_counter(offsetof(Counters, table_updates));
    flip_state_bit(model, output);
    stop_cpu_counter();
#endif
}

void handle_softmax(Model*, const ParameterInfo*[], ParameterInfo*, const Node*) {
    // Do nothing - softmax does not change the relative order of values.
    // Just let run_model determine the max value
}

void handle_transpose(Model*, const ParameterInfo *input[], ParameterInfo *output, const Node*) {
    my_printf_debug("Transpose!" NEWLINE);

    const ParameterInfo *X = input[0];
    // not actually transpose data as we happen to need NHWC
    // XXX: assume NHWC -> NCHW
    output->dims[1] = X->dims[3];
    output->dims[2] = X->dims[1];
    output->dims[3] = X->dims[2];
}

void alloc_add(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node *node) {
    output->slot = get_next_slot(model, input[0]);
}

void handle_add(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node *node) {
    my_printf_debug("Add!" NEWLINE);

    const ParameterInfo *X = input[0], *Y = input[1];

    uint32_t data_offset = 0;
#if INTERMITTENT
    start_cpu_counter(offsetof(Counters, progress_seeking));
    uint32_t first_unfinished_job_idx = run_recovery(model, output);
    data_offset = batch_start(job_index_to_offset(output, first_unfinished_job_idx));

#if INDIRECT_RECOVERY
    start_cpu_counter(offsetof(Counters, state_query));
    uint16_t next_input_turning_point, next_output_turning_point;
    int16_t input_offset, output_offset;
    uint8_t input_turning_point_idx, output_turning_point_idx;
    SlotInfo *input_slot_info, *output_slot_info;
    find_initial_state_bit(&input_offset, &input_turning_point_idx, &next_input_turning_point, &input_slot_info,
                           data_offset, model, X);
    find_initial_state_bit(&output_offset, &output_turning_point_idx, &next_output_turning_point, &output_slot_info,
                           data_offset, model, output);
    stop_cpu_counter();
#endif
    stop_cpu_counter();
#endif

    uint16_t buffer_size = X->dims[1];
    int16_t *buffer_a = lea_buffer,
            *buffer_b = buffer_a + buffer_size;
    my_memcpy_from_param(model, buffer_b, Y, 0, buffer_size * sizeof(int16_t));
#if JAPARI
    start_cpu_counter(offsetof(Counters, embedding));
    move_weights(buffer_b, false, extend_for_footprints(buffer_size), buffer_size);
    stop_cpu_counter();
#endif
    my_printf_debug("weights" NEWLINE);
    dump_matrix_debug(buffer_b, buffer_size, ValueInfo(Y), false);

    int16_t scaleFract;
    uint8_t shift;
    float_to_scale_params(&scaleFract, &shift, Y->scale/X->scale);
    my_scale_q15(buffer_b, scaleFract, shift, buffer_b, buffer_size);

    uint16_t idx = data_offset / buffer_size;
    uint16_t cur_buffer_size = buffer_size - (data_offset - idx * buffer_size);
    for (; idx < X->dims[2]; idx++) {
        my_printf_debug("data_offset=%d" NEWLINE, data_offset);
        my_memcpy_from_param(model, buffer_a, X, data_offset, cur_buffer_size * sizeof(int16_t));
#if STATEFUL
        my_printf_debug("Before strip states" NEWLINE);
        dump_matrix_debug(buffer_a, cur_buffer_size, ValueInfo(output), false);

        start_cpu_counter(offsetof(Counters, state_query));
        check_next_turning_point(input_offset, input_turning_point_idx, next_input_turning_point, input_slot_info, data_offset);
        stop_cpu_counter();

        start_cpu_counter(offsetof(Counters, embedding));
        update_states(buffer_a, cur_buffer_size, data_offset, input_offset, next_input_turning_point, false);
        stop_cpu_counter();

        my_printf_debug("After strip states" NEWLINE);
        dump_matrix_debug(buffer_a, cur_buffer_size, ValueInfo(output), false);
#endif

        my_add_q15(buffer_a, buffer_b + (buffer_size - cur_buffer_size), buffer_a, cur_buffer_size);
        my_printf_debug("After add" NEWLINE);
        dump_matrix_debug(buffer_a, cur_buffer_size, ValueInfo(output), false);

#if INDIRECT_RECOVERY
        start_cpu_counter(offsetof(Counters, state_query));
        check_next_turning_point(output_offset, output_turning_point_idx, next_output_turning_point, output_slot_info, data_offset);
        stop_cpu_counter();
        start_cpu_counter(offsetof(Counters, embedding));
        update_states(buffer_a, cur_buffer_size, data_offset, output_offset, next_output_turning_point, true);
        stop_cpu_counter();
        my_printf_debug("After embedding states" NEWLINE);
        dump_matrix_debug(buffer_a, cur_buffer_size, ValueInfo(output), true);
#endif

        my_memcpy_to_param(output, data_offset, buffer_a, cur_buffer_size * sizeof(int16_t), 0);
        data_offset += cur_buffer_size;
#if HAWAII
        write_hawaii_layer_footprint(model->layer_idx, cur_buffer_size/BATCH_SIZE*BATCH_SIZE);
#endif
        cur_buffer_size = buffer_size;
    }

#if INDIRECT_RECOVERY
    start_cpu_counter(offsetof(Counters, table_updates));
    flip_state_bit(model, output);
    stop_cpu_counter();
#endif

    dump_params_nhwc_debug(model, output, node->output_name);
}
