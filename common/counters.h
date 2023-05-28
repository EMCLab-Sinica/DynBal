#pragma once

#include "data.h"
#include "data_structures.h"
#include "my_debug.h"
#include "cnn_common.h"
#include <cstdint>

#define ENABLE_COUNTERS 1

// Counter pointers have the form offsetof(Counter, field_name). I use offsetof() instead of
// pointers to member fields like https://stackoverflow.com/questions/670734/pointer-to-class-data-member
// as the latter involves pointer arithmetic and is slower for platforms with special pointer bitwidths (ex: MSP430)
#if ENABLE_COUNTERS

extern uint8_t current_counter;
extern uint8_t prev_counter;
const uint8_t INVALID_POINTER = 0xff;

void load_counters(void);
void add_counter(uint8_t counter, uint32_t value);
uint32_t get_counter(uint8_t counter);
#if !ENABLE_DEMO_COUNTERS
void start_cpu_counter(uint8_t mem_ptr);
void stop_cpu_counter(void);
#else
#define start_cpu_counter(mem_ptr)
#define stop_cpu_counter()
#endif

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
