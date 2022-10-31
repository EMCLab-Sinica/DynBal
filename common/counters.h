#pragma once

#include "my_debug.h"
#include "cnn_common.h"
#include <cstdint>

#define ENABLE_COUNTERS 0
#define ENABLE_PER_LAYER_COUNTERS 0
#define ENABLE_DEMO_COUNTERS 0
// Some demo codes assume counters are accumulated across layers
static_assert((!ENABLE_PER_LAYER_COUNTERS) || (!ENABLE_DEMO_COUNTERS), "ENABLE_PER_LAYER_COUNTERS and ENABLE_DEMO_COUNTERS are mutually exclusive");

// Counter pointers have the form offsetof(Counter, field_name). I use offsetof() instead of
// pointers to member fields like https://stackoverflow.com/questions/670734/pointer-to-class-data-member
// as the latter involves pointer arithmetic and is slower for platforms with special pointer bitwidths (ex: MSP430)
#if ENABLE_COUNTERS

#define COUNTERS_LEN (MODEL_NODES_LEN+1)
struct Counters {
    // field offset = 0
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
    uint32_t nvm_write_linear_jobs;
    uint32_t nvm_write_non_linear_jobs;
    uint32_t nvm_write_footprints;
    uint32_t nvm_write_shadow_data;
    uint32_t nvm_write_model;
};

extern uint8_t counters_cur_copy_id;
extern Counters (*counters_data)[COUNTERS_LEN];
Counters *counters();
#if ENABLE_DEMO_COUNTERS
extern uint32_t total_jobs;
#endif

extern uint8_t current_counter;
extern uint8_t prev_counter;
const uint8_t INVALID_POINTER = 0xff;

static inline void add_counter(uint8_t counter, uint32_t value) {
    *reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(counters()) + counter) += value;
}

static inline void start_cpu_counter(uint8_t mem_ptr) {
#if ENABLE_DEMO_COUNTERS
    return;
#endif

    MY_ASSERT(prev_counter == INVALID_POINTER, "There is already two counters - prev_counter=%d, current_counter=%d", prev_counter, current_counter);

    if (current_counter != INVALID_POINTER) {
        prev_counter = current_counter;
        add_counter(prev_counter, plat_stop_cpu_counter());
        my_printf_debug("Stopping outer CPU counter %d" NEWLINE, prev_counter);
    }
    my_printf_debug("Start CPU counter %d" NEWLINE, mem_ptr);
    current_counter = mem_ptr;
    plat_start_cpu_counter();
}

static inline void stop_cpu_counter(void) {
#if ENABLE_DEMO_COUNTERS
    return;
#endif

    MY_ASSERT(current_counter != INVALID_POINTER);

    my_printf_debug("Stop inner CPU counter %d" NEWLINE, current_counter);
    add_counter(current_counter, plat_stop_cpu_counter());
    if (prev_counter != INVALID_POINTER) {
        current_counter = prev_counter;
        my_printf_debug("Restarting outer CPU counter %d" NEWLINE, current_counter);
        plat_start_cpu_counter();
        prev_counter = INVALID_POINTER;
    } else {
        current_counter = INVALID_POINTER;
    }
}

void print_all_counters();
void reset_counters();
bool counters_cleared();
void report_progress();

#else
#define start_cpu_counter(mem_ptr)
#define stop_cpu_counter()
#define print_all_counters()
#define reset_counters()
#define report_progress()
#endif

// A global switch for disabling counters temporarily
extern uint8_t counters_enabled;

static inline void enable_counters() {
    counters_enabled = 1;
}

static inline void disable_counters() {
    counters_enabled = 0;
}
