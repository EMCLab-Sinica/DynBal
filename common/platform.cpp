#include "platform.h"
#include "platform-private.h"
#include "cnn_common.h"
#include "my_debug.h"

// put offset checks here as extra headers are used
static_assert(INTERMEDIATE_PARAMETERS_INFO_OFFSET > NUM_SLOTS * INTERMEDIATE_VALUES_SIZE, "Incorrect NVM layout");
static_assert(MODEL_OFFSET > INTERMEDIATE_PARAMETERS_INFO_OFFSET + INTERMEDIATE_PARAMETERS_INFO_DATA_LEN, "Incorrect NVM layout");
static_assert(FIRST_RUN_OFFSET > MODEL_OFFSET + 2 * sizeof(Model), "Incorrect NVM layout");
static_assert(COUNTERS_OFFSET > FIRST_RUN_OFFSET + sizeof(uint8_t), "Incorrect NVM layout");

static uint32_t intermediate_values_offset(uint8_t slot_id) {
    return 0 + slot_id * INTERMEDIATE_VALUES_SIZE;
}

static uint32_t intermediate_parameters_info_addr(uint8_t i) {
    return INTERMEDIATE_PARAMETERS_INFO_OFFSET + i * sizeof(ParameterInfo);
}

static uint32_t model_addr(uint8_t i) {
    return MODEL_OFFSET + i * sizeof(Model);
}

void my_memcpy_to_param(struct ParameterInfo *param, uint16_t offset_in_word, const void *src, size_t n) {
    MY_ASSERT(param->bitwidth == 16);
    MY_ASSERT(param->slot < SLOT_CONSTANTS_MIN);
    uint32_t total_offset = param->params_offset + offset_in_word * sizeof(int16_t);
    MY_ASSERT(total_offset + n <= INTERMEDIATE_VALUES_SIZE);
    write_to_nvm(src, intermediate_values_offset(param->slot) + total_offset, n);
}

void my_memcpy_from_intermediate_values(void *dest, const ParameterInfo *param, uint16_t offset_in_word, size_t n) {
    read_from_nvm(dest, intermediate_values_offset(param->slot) + offset_in_word * sizeof(int16_t), n);
}

ParameterInfo* get_intermediate_parameter_info(uint8_t i) {
    ParameterInfo* dst = intermediate_parameters_info_vm + i;
    read_from_nvm(dst, intermediate_parameters_info_addr(i), sizeof(ParameterInfo));
    my_printf_debug("Load intermediate parameter info %d from NVM" NEWLINE, i);
    MY_ASSERT(dst->parameter_info_idx == i + N_INPUT);
    return dst;
}

void commit_intermediate_parameter_info(uint8_t i) {
    const ParameterInfo* src = intermediate_parameters_info_vm + i;
    MY_ASSERT(src->parameter_info_idx == i + N_INPUT);
    write_to_nvm(src, intermediate_parameters_info_addr(i), sizeof(ParameterInfo));
    my_printf_debug("Committing intermediate parameter info %d to NVM" NEWLINE, i);
}

static uint8_t get_newer_model_copy_id(void) {
    uint16_t version1, version2;
    read_from_nvm(&version1, model_addr(0) + offsetof(Model, version), sizeof(uint16_t));
    read_from_nvm(&version2, model_addr(1) + offsetof(Model, version), sizeof(uint16_t));
    my_printf_debug("Versions of shadow Model copies: %d, %d" NEWLINE, version1, version2);

    if (abs(static_cast<int>(version1 - version2)) == 1) {
        if (version1 > version2) {
            return 0;
        } else {
            return 1;
        }
    } else {
        if (version1 > version2) {
            // ex: versions = 65535, 1
            return 1;
        } else {
            return 0;
        }
    }
}

Model* get_model(void) {
    Model *dst = &model_vm;

    uint8_t newer_model_copy_id = get_newer_model_copy_id();
    read_from_nvm(dst, model_addr(newer_model_copy_id), sizeof(Model));
    my_printf_debug("Using model copy %d, version %d" NEWLINE, newer_model_copy_id, dst->version);
    return dst;
}

void commit_model(void) {
    uint8_t newer_model_copy_id = get_newer_model_copy_id();
    uint8_t older_model_copy_id = newer_model_copy_id ^ 1;

    bump_model_version(&model_vm);

    write_to_nvm(&model_vm, model_addr(older_model_copy_id), sizeof(Model));
    my_printf_debug("Committing version %d to model copy %d" NEWLINE, model_vm.version, older_model_copy_id);
}

void first_run(void) {
#if STATEFUL_CNN
    my_erase(intermediate_values_offset(0), INTERMEDIATE_VALUES_SIZE * NUM_SLOTS);
#endif
    write_to_nvm(intermediate_parameters_info_data, intermediate_parameters_info_addr(0), INTERMEDIATE_PARAMETERS_INFO_DATA_LEN);
    write_to_nvm(model_data, model_addr(0), MODEL_DATA_LEN);
    write_to_nvm(model_data, model_addr(1), MODEL_DATA_LEN);

    get_model(); // refresh model_vm
    commit_model();
}
