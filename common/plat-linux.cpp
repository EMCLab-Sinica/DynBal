#ifdef POSIX_BUILD

#define _POSIX_C_SOURCE 1 // for kill()

#include "intermittent-cnn.h"
#include "cnn_common.h"
#include "my_debug.h"
#include "platform.h"
#include "data.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <getopt.h>
#include <unistd.h>
#include <signal.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/ptrace.h>

#define MEMCPY_DELAY_US 0

/* data on NVM, made persistent via mmap() with a file */
uint8_t *nvm;
uint16_t dma_invocations[COUNTERS_LEN];
uint16_t dma_bytes[COUNTERS_LEN];

Counters *counters() {
    return reinterpret_cast<Counters*>(nvm + COUNTERS_OFFSET);
}

int main(int argc, char* argv[]) {
    int ret = 0, opt_ch, button_pushed = 0, read_only = 0, n_samples = 0;
    Model *model;
    int nvm_fd = -1, samples_fd = -1;

    while((opt_ch = getopt(argc, argv, "bfr")) != -1) {
        switch (opt_ch) {
            case 'b':
                button_pushed = 1;
                break;
            case 'r':
                read_only = 1;
                break;
            case 'f':
                dump_integer = 0;
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

    struct stat stat_buf;
    if (stat("nvm.bin", &stat_buf) != 0) {
        if (errno != ENOENT) {
            perror("Checking nvm.bin failed");
            goto exit;
        }
        nvm_fd = open("nvm.bin", O_RDWR|O_CREAT, 0600);
        ftruncate(nvm_fd, NVM_SIZE);
    } else {
        nvm_fd = open("nvm.bin", O_RDWR);
    }
    nvm = reinterpret_cast<uint8_t*>(mmap(NULL, NVM_SIZE, PROT_READ|PROT_WRITE, read_only ? MAP_PRIVATE : MAP_SHARED, nvm_fd, 0));
    if (nvm == MAP_FAILED) {
        perror("mmap() failed");
        goto exit;
    }

    samples_fd = open("samples.bin", O_RDONLY);
    samples_data = reinterpret_cast<uint8_t*>(mmap(NULL, SAMPLE_SIZE * N_SAMPLES, PROT_READ, MAP_PRIVATE, samples_fd, 0));
    if (samples_data == MAP_FAILED) {
        perror("mmap() for samples failed");
        goto exit;
    }

#ifdef USE_ARM_CMSIS
    my_printf("Use DSP from ARM CMSIS pack" NEWLINE);
#else
    my_printf("Use TI DSPLib" NEWLINE);
#endif

    model = get_model();

    // emulating button_pushed - treating as a fresh run
    if (button_pushed) {
        model->version = 0;
    }

    if (!model->version) {
        // the first time
        first_run();
    }

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

void setOutputValue(uint8_t) {}

void my_memcpy(void* dest, const void* src, size_t n) {
    uint16_t counter_idx = counters()->counter_idx;
    dma_invocations[counter_idx]++;
    dma_bytes[counter_idx] += n;
#if MEMCPY_DELAY_US
    usleep(MEMCPY_DELAY_US);
#endif
    // my_printf_debug("%s copied %zu bytes" NEWLINE, __func__, n);
    // Not using memcpy here so that it is more likely that power fails during
    // memcpy, which is the case for external FRAM
    uint8_t *dest_u = reinterpret_cast<uint8_t*>(dest);
    const uint8_t *src_u = reinterpret_cast<const uint8_t*>(src);
    for (size_t idx = 0; idx < n; idx++) {
        dest_u[idx] = src_u[idx];
    }
}

void read_from_nvm(void *vm_buffer, uint32_t nvm_offset, size_t n) {
    my_memcpy(vm_buffer, nvm + nvm_offset, n);
}

void write_to_nvm(const void *vm_buffer, uint32_t nvm_offset, size_t n) {
    my_memcpy(nvm + nvm_offset, vm_buffer, n);
}

void my_erase(uint32_t nvm_offset, size_t n) {
    memset(nvm + nvm_offset, 0, n);
}

[[ noreturn ]] void ERROR_OCCURRED(void) {
    if (ptrace(PTRACE_TRACEME, 0, NULL, 0) == -1) {
        // Let the debugger break
        kill(getpid(), SIGINT);
    }
    // give up otherwise
    exit(1);
}

#endif // POSIX_BUILD
