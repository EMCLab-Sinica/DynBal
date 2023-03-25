#include <cinttypes>
#include <cstdint>
#include <cstring>
#include "cnn_common.h"
#include "counters.h"
#include "platform.h"

uint8_t counters_enabled = 1;

#if ENABLE_COUNTERS
uint8_t current_counter = INVALID_POINTER;
uint8_t prev_counter = INVALID_POINTER;

Counters *counters() {
#if ENABLE_PER_LAYER_COUNTERS
    return counters_data[counters_cur_copy_id] + model_vm.layer_idx;
#else
    return counters_data[counters_cur_copy_id];
#endif
}

template<uint32_t Counters::* MemPtr>
static uint32_t print_counters() {
    uint32_t total = 0;
    for (uint16_t i = 0; i < MODEL_NODES_LEN; i++) {
        total += counters_data[counters_cur_copy_id][i].*MemPtr;
#if ENABLE_PER_LAYER_COUNTERS
        my_printf("%12" PRIu32, counters_data[counters_cur_copy_id][i].*MemPtr);
#else
        break;
#endif
    }
    my_printf(" total=%12" PRIu32, total);
    return total;
}

void print_all_counters() {
#if ENABLE_DEMO_COUNTERS
    return;
#endif

    my_printf("op types:                ");
#if ENABLE_PER_LAYER_COUNTERS
    for (uint16_t i = 0; i < MODEL_NODES_LEN; i++) {
        my_printf("% 12d", get_node(i)->op_type);
    }
#endif
    uint32_t total_dma_bytes = 0, total_macs = 0, total_overhead = 0;
    my_printf(NEWLINE "Power counters:          "); print_counters<&Counters::power_counters>();
    my_printf(NEWLINE "MACs:                    "); total_macs = print_counters<&Counters::macs>();
    // state-embedding overheads
    my_printf(NEWLINE "Embeddings:              "); total_overhead += print_counters<&Counters::embedding>();
    my_printf(NEWLINE "Strippings:              "); total_overhead += print_counters<&Counters::stripping>();
    my_printf(NEWLINE "Overflow handling:       "); total_overhead += print_counters<&Counters::overflow_handling>();
    // state-assignment overheads
    my_printf(NEWLINE "State queries:           "); total_overhead += print_counters<&Counters::state_query>();
    my_printf(NEWLINE "Table updates:           "); total_overhead += print_counters<&Counters::table_updates>();
    my_printf(NEWLINE "Table preservation:      "); total_overhead += print_counters<&Counters::table_preservation>();
    my_printf(NEWLINE "Table loading:           "); total_overhead += print_counters<&Counters::table_loading>();
    // recovery overheads
    my_printf(NEWLINE "Progress seeking:        "); total_overhead += print_counters<&Counters::progress_seeking>();
    // misc
    my_printf(NEWLINE "Memory layout:           "); total_overhead += print_counters<&Counters::memory_layout>();
#if JAPARI
    my_printf(NEWLINE "Data loading:            "); total_overhead += print_counters<&Counters::data_loading>();
#endif
#if STATEFUL
    my_printf(NEWLINE "Embedded values:         "); total_overhead += print_counters<&Counters::embedded_values>();
#endif
    my_printf(NEWLINE "DMA invocations:         "); print_counters<&Counters::dma_invocations>();
    my_printf(NEWLINE "DMA bytes:               "); total_dma_bytes = print_counters<&Counters::dma_bytes>();
    my_printf(NEWLINE "DMA (VM to VM):          "); print_counters<&Counters::dma_vm_to_vm>();
    my_printf(NEWLINE "NVM read (job outputs):  "); print_counters<&Counters::nvm_read_job_outputs>();
    my_printf(NEWLINE "NVM read (parameters):   "); print_counters<&Counters::nvm_read_parameters>();
    my_printf(NEWLINE "NVM read (shadow data):  "); print_counters<&Counters::nvm_read_shadow_data>();
    my_printf(NEWLINE "NVM read (model data):   "); print_counters<&Counters::nvm_read_model>();
    my_printf(NEWLINE "NVM write (L jobs):      "); total_overhead += print_counters<&Counters::nvm_write_linear_jobs>();
    my_printf(NEWLINE "NVM write (NL jobs):     "); total_overhead += print_counters<&Counters::nvm_write_non_linear_jobs>();
    my_printf(NEWLINE "NVM write (footprints):  "); total_overhead += print_counters<&Counters::nvm_write_footprints>();
    my_printf(NEWLINE "NVM write (shadow data): "); print_counters<&Counters::nvm_write_shadow_data>();
    my_printf(NEWLINE "NVM write (model data):  "); print_counters<&Counters::nvm_write_model>();

    my_printf(NEWLINE "Total DMA bytes: %d", total_dma_bytes);
    my_printf(NEWLINE "Total MACs: %d", total_macs);
    my_printf(NEWLINE "Communication-to-computation ratio: %f", 1.0f*total_dma_bytes/total_macs);
    my_printf(NEWLINE "Total overhead: %" PRIu32, total_overhead);
    my_printf(NEWLINE "run_counter: %d" NEWLINE, get_model()->run_counter);
}

static uint32_t* get_counter_ptr(uint8_t counter) {
    return reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(counters()) + counter);
}

void add_counter(uint8_t counter, uint32_t value) {
    *get_counter_ptr(counter) += value;
}

uint32_t get_counter(uint8_t counter) {
    return *get_counter_ptr(counter);
}

void reset_counters() {
#if ENABLE_COUNTERS
    memset(counters_data[counters_cur_copy_id ^ 1], 0, sizeof(Counters) * COUNTERS_LEN);
    counters_cur_copy_id ^= 1;
#endif
}

bool counters_cleared() {
    return (current_counter == INVALID_POINTER) && (prev_counter == INVALID_POINTER);
}

void start_cpu_counter(uint8_t mem_ptr) {
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

void stop_cpu_counter(void) {
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

void report_progress() {
#if ENABLE_DEMO_COUNTERS
    static uint8_t last_progress = 0;

    if (!total_jobs) {
        return;
    }
    uint32_t cur_jobs = (get_counter(offsetof(Counters, nvm_write_linear_jobs)) + get_counter(offsetof(Counters, nvm_write_non_linear_jobs))) / 2;
    uint8_t cur_progress = 100 * cur_jobs / total_jobs;
    // report only when the percentage is changed to avoid high UART overheads
    if (cur_progress != last_progress) {
        my_printf("P,%d,%d,", cur_progress,
                  cur_jobs/1024);
        my_printf("%d" NEWLINE, get_counter(offsetof(Counters, nvm_write_footprints))/1024);
        last_progress = cur_progress;
    }
#endif
}

#endif
