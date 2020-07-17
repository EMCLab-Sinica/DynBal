#include "intermittent-cnn.h"
#include "cnn_common.h"
#include "debug.h"
#include "platform.h"
#include "data.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <getopt.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>

#define MEMCPY_DELAY_US 0

/* data on NVM, made persistent via mmap() with a file */
uint8_t *nvm;
uint8_t *parameters_data, *parameters2_data, *samples_data, *model_data, *labels_data;
uint16_t dma_invocations[COUNTERS_LEN];
uint16_t dma_bytes[COUNTERS_LEN];

uint8_t *intermediate_values(uint8_t slot_id) {
    return nvm + CACHED_FILTERS_LEN + slot_id * INTERMEDIATE_VALUES_SIZE;
}

Counters *counters() {
    return (Counters*)(labels_data + LABELS_DATA_LEN);
}

int main(int argc, char* argv[]) {
    int ret = 0, opt_ch, read_only = 0, n_samples = 0;

    while((opt_ch = getopt(argc, argv, "r")) != -1) {
        switch (opt_ch) {
            case 'r':
                read_only = 1;
                break;
            default:
                printf("Usage: %s [-r] [n_samples]\n", argv[0]);
                return 1;
        }
    }
    if (argv[optind]) {
        n_samples = atoi(argv[optind]);
    }

    chdir(MY_SOURCE_DIR);

    int nvm_fd = open("nvm.bin", O_RDWR);
    nvm = mmap(NULL, NVM_SIZE, PROT_READ|PROT_WRITE, read_only ? MAP_PRIVATE : MAP_SHARED, nvm_fd, 0);
    if (nvm == MAP_FAILED) {
        perror("mmap() failed");
        goto exit;
    }
    // Keep the order consistent with `outputs` in transform.py
    parameters_data = intermediate_values(NUM_SLOTS);
    parameters2_data = parameters_data + PARAMETERS_DATA_LEN;
    samples_data = parameters2_data + PARAMETERS2_DATA_LEN;
    model_data = samples_data + SAMPLES_DATA_LEN;
    labels_data = model_data + MODEL_DATA_LEN;

#ifdef USE_ARM_CMSIS
    my_printf("Use DSP from ARM CMSIS pack" NEWLINE);
#else
    my_printf("Use TI DSPLib" NEWLINE);
#endif

    ret = run_cnn_tests(n_samples);

    for (uint16_t counter_idx = 0; counter_idx < COUNTERS_LEN; counter_idx++) {
        dma_invocations[counter_idx] = 0;
        dma_bytes[counter_idx] = 0;
    }

exit:
    close(nvm_fd);
    return ret;
}

void plat_reset_model(void) {
}

void plat_print_results(void) {
    my_printf(NEWLINE "DMA invocations:" NEWLINE);
    for (uint8_t i = 0; i < counters()->counter_idx; i++) {
        my_printf("% 8d", dma_invocations[i]);
    }
    my_printf(NEWLINE "DMA bytes:" NEWLINE);
    for (uint8_t i = 0; i < counters()->counter_idx; i++) {
        my_printf("% 8d", dma_bytes[i]);
    }
}

void setOutputValue(uint8_t value) {
    my_printf("Output set to %d" NEWLINE, value);
}

void my_memcpy(void* dest, const void* src, size_t n) {
    uint16_t counter_idx = counters()->counter_idx;
    dma_invocations[counter_idx]++;
    dma_bytes[counter_idx] += n;
#if MEMCPY_DELAY_US
    usleep(MEMCPY_DELAY_US);
#endif
    my_printf_debug(__func__);
    my_printf_debug(" copied %d bytes" NEWLINE, (int)n);
    memcpy(dest, src, n);
}

void fill_int16(int16_t *dest, uint16_t n, int16_t val) {
    for (size_t idx = 0; idx < n; idx++) {
        dest[idx] = val;
    }
}

_Noreturn void ERROR_OCCURRED(void) {
    abort();
}
