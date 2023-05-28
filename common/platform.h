#pragma once

#include <cstdint>
#include <cstdlib>

#if defined(__MSP430__) || defined(__MSP432__)
#  include "plat-mcu.h"
#else
#  include "plat-pc.h"
#endif

/* offsets for data on NVM */

// growing up (like heap). Not starting from zero as first few 16 bytes are for testing (see testSPI() function)
#define LAST_REPORTED_LAYER_IDX_OFFSET 16
#define INTERMEDIATE_VALUES_OFFSET 256
#define SAMPLES_OFFSET (INTERMEDIATE_VALUES_OFFSET + NUM_SLOTS * INTERMEDIATE_VALUES_SIZE)
#define PARAMETERS_OFFSET (SAMPLES_OFFSET + SAMPLES_DATA_LEN)

// growing down (like stack)
#define INFERENCE_STATS_OFFSET (NVM_SIZE - INFERENCE_STATS_DATA_LEN)
#define MODEL_OFFSET (INFERENCE_STATS_OFFSET - 2 * MODEL_DATA_LEN)
#define INTERMEDIATE_PARAMETERS_INFO_OFFSET (MODEL_OFFSET - INTERMEDIATE_PARAMETERS_INFO_DATA_LEN)
// Generated node_flags_data already contains two copies for shadowing, so there is no need to multiply the size by 2 here
#define NODE_FLAGS_OFFSET (INTERMEDIATE_PARAMETERS_INFO_OFFSET - NODE_FLAGS_DATA_LEN)
#define NODES_OFFSET (NODE_FLAGS_OFFSET - NODES_DATA_LEN)
#define FOOTPRINTS_OFFSET (NODES_OFFSET - FOOTPRINTS_DATA_LEN)
#define COUNTERS_OFFSET (FOOTPRINTS_OFFSET - COUNTERS_DATA_LEN)

#define DISABLE_FEATURE_MAP_NVM_ACCESS 0

struct ParameterInfo;
struct Model;
struct Counters;
enum class GPIOFlag;
extern Model model_vm;

[[ noreturn ]] void ERROR_OCCURRED(void);
void read_from_nvm(void* vm_buffer, uint32_t nvm_offset, size_t n);
void write_to_nvm(const void* vm_buffer, uint32_t nvm_offset, size_t n, uint16_t timer_delay = 0);
// DMA controller on MSP432 can handle at most 1024 words at a time
void read_from_nvm_segmented(uint8_t* vm_buffer, uint32_t nvm_offset, uint32_t total_len, uint16_t segment_size);
void write_to_nvm_segmented(const uint8_t* vm_buffer, uint32_t nvm_offset, uint32_t total_len, uint16_t segment_size = 1024);
void my_erase(void);
void copy_data_to_nvm(void);
void my_memcpy(void* dest, const void* src, size_t n);
void my_memcpy_to_param(ParameterInfo *param, uint16_t offset_in_word, const void *src, size_t n, uint16_t timer_delay, bool is_linear);
void my_memcpy_from_intermediate_values(void *dest, const ParameterInfo *param, uint16_t offset_in_word, size_t n);
// offset_in_bytes may go beyond 64K after being multiplied with sizeof(T)
void my_memcpy_from_parameters(void *dest, const ParameterInfo *param, uint32_t offset_in_bytes, size_t n);
void read_from_samples(void *dest, uint16_t offset_in_word, size_t n);
ParameterInfo* get_intermediate_parameter_info(uint16_t i);
void commit_intermediate_parameter_info(uint16_t i);
Model* get_model(void);
Model* load_model_from_nvm(void);
void commit_model(void);
void first_run(void);
void notify_layer_finished(void);
void notify_model_finished(void);
void notify_indicator(uint8_t idx);
bool read_gpio_flag(GPIOFlag flag);
#if HAWAII
void write_hawaii_layer_footprint(uint16_t layer_idx, int16_t n_jobs);
uint16_t read_hawaii_layer_footprint(uint16_t layer_idx);
void reset_hawaii_layer_footprint(uint16_t layer_idx);
#endif
