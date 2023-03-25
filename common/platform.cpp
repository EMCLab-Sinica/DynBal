#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include "c_callbacks.h"
#include "counters.h"
#include "data.h"
#include "platform.h"
#include "cnn_common.h"
#include "my_debug.h"
#include "intermittent-cnn.h" // for sample_idx
#include "double_buffering.h"

// put offset checks here as extra headers are used
static_assert(FOOTPRINTS_OFFSET >= PARAMETERS_OFFSET + PARAMETERS_DATA_LEN, "Incorrect NVM layout");

Model model_vm;

static uint32_t intermediate_values_offset(uint8_t slot_id) {
    return INTERMEDIATE_VALUES_OFFSET + slot_id * INTERMEDIATE_VALUES_SIZE;
}

static uint32_t intermediate_parameters_info_addr(uint16_t i) {
    return INTERMEDIATE_PARAMETERS_INFO_OFFSET + i * sizeof(ParameterInfo);
}

template<>
uint32_t nvm_addr<Model>(uint8_t i, uint16_t) {
    return MODEL_OFFSET + i * sizeof(Model);
}

template<>
Model* vm_addr<Model>(uint16_t data_idx) {
    return &model_vm;
}

template<>
const char* datatype_name<Model>(void) {
    return "model";
}

static void notify_progress(void) {
#if 0
    // indicate there is some progress in this power cycle
    static bool notified = false;
    if (!notified) {
        notify_indicator(1);
        notified = true;
    }
#endif
}

void my_memcpy_to_param(ParameterInfo *param, uint16_t offset_in_word, const void *src, size_t n, uint16_t timer_delay, bool is_linear) {
    MY_ASSERT(param->slot < NUM_SLOTS);
    uint32_t total_offset = param->params_offset + offset_in_word * sizeof(int16_t);
    MY_ASSERT(total_offset + n <= param->params_len);
#if ENABLE_COUNTERS
    uint32_t n_jobs;
#if JAPARI
    uint16_t n_footprints = n / (BATCH_SIZE + 1);
    n_jobs = n - n_footprints;
    add_counter(offsetof(Counters, nvm_write_footprints), n_footprints);
    my_printf_debug("Recorded %u bytes of footprints written to NVM" NEWLINE, n_footprints);
#else
    n_jobs = n;
#endif // JAPARI
    if (is_linear) {
        add_counter(offsetof(Counters, nvm_write_linear_jobs), n_jobs);
        my_printf_debug("Recorded %u bytes of linear jobs written to NVM" NEWLINE, n_jobs);
    } else {
        add_counter(offsetof(Counters, nvm_write_non_linear_jobs), n_jobs);
        my_printf_debug("Recorded %u bytes of non-linear jobs written to NVM" NEWLINE, n_jobs);
    }
#endif
    write_to_nvm(src, intermediate_values_offset(param->slot) + total_offset, n, timer_delay);
    notify_progress();
}

void my_memcpy_from_intermediate_values(void *dest, const ParameterInfo *param, uint16_t offset_in_word, size_t n) {
#if ENABLE_COUNTERS
    if (counters_enabled) {
        add_counter(offsetof(Counters, nvm_read_job_outputs), n);
        my_printf_debug("Recorded %lu bytes of job outputs fetched from NVM, accumulated=%" PRIu32 NEWLINE, n, get_counter(offsetof(Counters, nvm_read_job_outputs)));
    }
#endif
    read_from_nvm(dest, intermediate_values_offset(param->slot) + offset_in_word * sizeof(int16_t), n);
}

void read_from_samples(void *dest, uint16_t offset_in_word, size_t n) {
#if ENABLE_COUNTERS
    if (counters_enabled) {
        add_counter(offsetof(Counters, nvm_read_job_outputs), n);
        my_printf_debug("Recorded %lu bytes of samples fetched from NVM, accumulated=%" PRIu32 NEWLINE, n, get_counter(offsetof(Counters, nvm_read_job_outputs)));
    }
#endif
    read_from_nvm(dest, SAMPLES_OFFSET + (sample_idx % LABELS_DATA_LEN) * 2*TOTAL_SAMPLE_SIZE + offset_in_word * sizeof(int16_t), n);
}

ParameterInfo* get_intermediate_parameter_info(uint16_t i) {
#if ENABLE_COUNTERS
    if (counters_enabled) {
        add_counter(offsetof(Counters, nvm_read_model), sizeof(ParameterInfo));
        my_printf_debug("Recorded %lu bytes of ParameterInfo fetched from NVM" NEWLINE, sizeof(ParameterInfo));
    }
#endif
    ParameterInfo* dst = intermediate_parameters_info_vm + i;
    read_from_nvm(dst, intermediate_parameters_info_addr(i), sizeof(ParameterInfo));
    my_printf_debug("Load intermediate parameter info %d from NVM" NEWLINE, i);
    MY_ASSERT(dst->parameter_info_idx == i + N_INPUT,
              "Expect parameter index %d but got %d" NEWLINE, i + N_INPUT, dst->parameter_info_idx);
    return dst;
}

void commit_intermediate_parameter_info(uint16_t i) {
#if ENABLE_COUNTERS
    if (counters_enabled) {
        add_counter(offsetof(Counters, nvm_write_model), sizeof(ParameterInfo));
        my_printf_debug("Recorded %lu bytes of ParameterInfo written NVM" NEWLINE, sizeof(ParameterInfo));
    }
#endif
    const ParameterInfo* src = intermediate_parameters_info_vm + i;
    MY_ASSERT(src->parameter_info_idx == i + N_INPUT);
    write_to_nvm(src, intermediate_parameters_info_addr(i), sizeof(ParameterInfo));
    my_printf_debug("Committing intermediate parameter info %d to NVM" NEWLINE, i);
}

Model* load_model_from_nvm(void) {
    start_cpu_counter(offsetof(Counters, table_loading));
    Model* ret = get_versioned_data<Model>(0);
    stop_cpu_counter();
    return ret;
}

Model* get_model(void) {
    return &model_vm;
}

void commit_model(void) {
#if ENABLE_DEMO_COUNTERS
    if (!model_vm.running) {
        reset_counters();
    }
#endif
    start_cpu_counter(offsetof(Counters, table_preservation));
    commit_versioned_data<Model>(0);
    // send finish signals only after the whole network has really finished
#if ENABLE_COUNTERS
    add_counter(offsetof(Counters, power_counters), 1);
#endif
    if (!model_vm.running) {
        notify_model_finished();
    }
    stop_cpu_counter();
}

void first_run(void) {
    my_printf_debug("First run, resetting everything..." NEWLINE);
    disable_counters();
    my_erase();
    copy_data_to_nvm();
    reset_counters();

    write_to_nvm_segmented(intermediate_parameters_info_data, intermediate_parameters_info_addr(0),
                           INTERMEDIATE_PARAMETERS_INFO_DATA_LEN, sizeof(ParameterInfo));
    write_to_nvm(model_data, nvm_addr<Model>(0, 0), MODEL_DATA_LEN);
    write_to_nvm(model_data, nvm_addr<Model>(1, 0), MODEL_DATA_LEN);

    load_model_from_nvm(); // refresh model_vm
    commit_model();

    my_printf_debug("Init for " CONFIG "/" METHOD " with batch size=%d" NEWLINE, BATCH_SIZE);
    enable_counters();
}

void write_to_nvm_segmented(const uint8_t* vm_buffer, uint32_t nvm_offset, uint32_t total_len, uint16_t segment_size) {
    for (uint32_t idx = 0; idx < total_len; idx += segment_size) {
        write_to_nvm(vm_buffer + idx, nvm_offset + idx, MIN_VAL(total_len - idx, segment_size));
    }
}

void record_overflow_handling_overhead(uint32_t cycles) {
#if ENABLE_COUNTERS
    add_counter(offsetof(Counters, overflow_handling), cycles);
#endif
}

#if HAWAII
Footprint footprints_vm[MODEL_NODES_LEN];

template<>
uint32_t nvm_addr<Footprint>(uint8_t copy_id, uint16_t layer_idx) {
    return FOOTPRINTS_OFFSET + (copy_id * MODEL_NODES_LEN + layer_idx) * sizeof(Footprint);
}

template<>
Footprint* vm_addr<Footprint>(uint16_t layer_idx) {
    return &footprints_vm[layer_idx];
}

template<>
const char* datatype_name<Footprint>(void) {
    return "footprint";
}

void write_hawaii_layer_footprint(uint16_t layer_idx, int16_t n_jobs) {
    Footprint* footprint_vm = footprints_vm + layer_idx;
    footprint_vm->value += n_jobs;
    MY_ASSERT(footprint_vm->value < INTERMEDIATE_VALUES_SIZE);
    commit_versioned_data<Footprint>(layer_idx);
    my_printf_debug("Write HAWAII layer footprint %d for layer %d" NEWLINE, footprint_vm->value, layer_idx);
    MY_ASSERT(footprint_vm->value % BATCH_SIZE == 0);
}

uint16_t read_hawaii_layer_footprint(uint16_t layer_idx) {
    uint16_t footprint = get_versioned_data<Footprint>(layer_idx)->value;
    my_printf_debug("HAWAII layer footprint=%d for layer %d" NEWLINE, footprint, layer_idx);
    MY_ASSERT(footprint % BATCH_SIZE == 0);
    return footprint;
}

void reset_hawaii_layer_footprint(uint16_t layer_idx) {
    Footprint footprint;
    footprint.value = footprint.version = 0;
    write_to_nvm(&footprint, nvm_addr<Footprint>(0, layer_idx), sizeof(Footprint));
    write_to_nvm(&footprint, nvm_addr<Footprint>(1, layer_idx), sizeof(Footprint));
    my_printf_debug("Reset HAWAII layer footprint for layer %d" NEWLINE, layer_idx);
}
#endif
