#ifdef PC_BUILD

#include "intermittent-cnn.h"
#include "cnn_common.h"
#include "counters.h"
#include "my_debug.h"
#include "platform.h"
#include "data.h"
#include <cinttypes>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#ifdef __linux__
#include <fcntl.h>
#include <getopt.h>
#include <unistd.h>
#include <signal.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/ptrace.h>
#endif
#include <fstream>
#include <memory>
#ifdef USE_PROTOBUF
#include "model_output.pb.h"
#endif

/* data on NVM, made persistent via mmap() with a file */
uint8_t *nvm;
static uint32_t shutdown_counter = UINT32_MAX;
static std::ofstream out_file;

#if ENABLE_COUNTERS
Counters (*counters_data)[COUNTERS_LEN];
uint8_t counters_cur_copy_id = 0;
uint32_t total_jobs = 0;
#endif

#ifdef USE_PROTOBUF
static void save_model_output_data() {
    model_output_data->SerializeToOstream(&out_file);
}
#endif

static void* map_file(const char* path, size_t len, bool read_only) {
    int fd = -1;
    struct stat stat_buf;
    if (stat(path, &stat_buf) != 0) {
        if (errno != ENOENT) {
            perror("Checking file failed");
            return NULL;
        }
        fd = open(path, O_RDWR|O_CREAT, 0600);
    } else {
        fd = open(path, O_RDWR);
    }
    ftruncate(fd, len);
    void* ptr = mmap(NULL, len, PROT_READ|PROT_WRITE, read_only ? MAP_PRIVATE : MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
        perror("mmap() failed");
        return NULL;
    }
    return ptr;
}

int main(int argc, char* argv[]) {
    int ret = 0, opt_ch, read_only = 0, n_samples = 0;
    Model *model;

#ifdef __linux__

    while((opt_ch = getopt(argc, argv, "frc:s:")) != -1) {
        switch (opt_ch) {
            case 'r':
                read_only = 1;
                break;
            case 'f':
                dump_integer = 0;
                break;
            case 'c':
                shutdown_counter = atol(optarg);
                break;
            case 's':
#if defined(USE_PROTOBUF) && MY_DEBUG >= MY_DEBUG_LAYERS
                out_file.open(optarg);
                break;
#else
                my_printf("Cannot save outputs as protobuf support is not compiled or debug is not enabled." NEWLINE);
                return 1;
#endif
            default:
                my_printf("Usage: %s [-r] [n_samples]" NEWLINE, argv[0]);
                return 1;
        }
    }
    if (argv[optind]) {
        n_samples = atoi(argv[optind]);
    }

    nvm = reinterpret_cast<uint8_t*>(map_file("nvm.bin", NVM_SIZE, read_only));
#if ENABLE_COUNTERS
    counters_data = reinterpret_cast<Counters(*)[COUNTERS_LEN]>(map_file("counters.bin", 2*COUNTERS_LEN*sizeof(Counters), read_only));
#endif

#else
    (void)read_only; // no simulated NVM other than Linux - silent a compiler warning
    nvm = new uint8_t[NVM_SIZE]();
#endif

#if USE_ARM_CMSIS
    my_printf_debug("Use DSP from ARM CMSIS pack" NEWLINE);
#else
    my_printf_debug("Use TI DSPLib" NEWLINE);
#endif

#ifdef USE_PROTOBUF
    if (out_file.is_open()) {
        model_output_data = std::make_unique<ModelOutput>();
    }
#endif

    model = load_model_from_nvm();

    if (!model->version) {
        // the first time
        first_run();
    }

    ret = run_cnn_tests(n_samples);

#if ENABLE_COUNTERS
    print_all_counters();
#endif

#ifdef USE_PROTOBUF
    if (out_file.is_open()) {
        save_model_output_data();
    }
#endif

#ifndef __linux__
    delete [] nvm;
#endif
    return ret;
}

[[ noreturn ]] static void exit_with_status(uint8_t exit_code) {
#ifdef __linux__
    if (ptrace(PTRACE_TRACEME, 0, NULL, 0) == -1) {
        // Let the debugger break
        kill(getpid(), SIGTERM);
    }
#endif
    // give up otherwise
    exit(exit_code);
}

void my_memcpy_ex(void* dest, const void* src, size_t n, uint8_t write_to_nvm) {
#if ENABLE_COUNTERS
    if (counters_enabled) {
        add_counter(offsetof(Counters, dma_invocations), 1);
        add_counter(offsetof(Counters, dma_bytes), n);
        my_printf_debug("Recorded %lu DMA bytes" NEWLINE, n);
    }
#endif
    // Not using memcpy here so that it is more likely that power fails during
    // memcpy, which is the case for external FRAM
    uint8_t *dest_u = reinterpret_cast<uint8_t*>(dest);
    const uint8_t *src_u = reinterpret_cast<const uint8_t*>(src);
    for (size_t idx = 0; idx < n; idx++) {
        dest_u[idx] = src_u[idx];
        if (write_to_nvm && counters_enabled) {
            shutdown_counter--;
            if (!shutdown_counter) {
                exit_with_status(2);
            }
        }
    }
}

void my_memcpy(void* dest, const void* src, size_t n) {
#if ENABLE_COUNTERS
    if (counters_enabled) {
        add_counter(offsetof(Counters, dma_vm_to_vm), n);
        my_printf_debug("Recorded %lu bytes copied from VM to VM" NEWLINE, n);
    }
#endif
    my_memcpy_ex(dest, src, n, 0);
}

void my_memcpy_from_parameters(void *dest, const ParameterInfo *param, uint32_t offset_in_bytes, size_t n) {
    MY_ASSERT(offset_in_bytes + n <= PARAMETERS_DATA_LEN);
#if ENABLE_COUNTERS
    if (counters_enabled) {
        add_counter(offsetof(Counters, nvm_read_parameters), n);
        my_printf_debug("Recorded %lu bytes fetched from parameters, accumulated=%" PRIu32 NEWLINE, n, get_counter(offsetof(Counters, nvm_read_parameters)));
    }
#endif
    my_memcpy_ex(dest, parameters_data + param->params_offset + offset_in_bytes, n, 0);
}

void read_from_nvm(void *vm_buffer, uint32_t nvm_offset, size_t n) {
    MY_ASSERT(n <= 1024);
    my_memcpy_ex(vm_buffer, nvm + nvm_offset, n, 0);
}

void write_to_nvm(const void *vm_buffer, uint32_t nvm_offset, size_t n, uint16_t timer_delay) {
    MY_ASSERT(n <= 1024);
    check_nvm_write_address(nvm_offset, n);
    my_memcpy_ex(nvm + nvm_offset, vm_buffer, n, 1);
}

void my_erase() {
    memset(nvm, 0, NVM_SIZE);
}

void copy_data_to_nvm(void) {
    // samples data
    std::ifstream samples_file("samples.bin", std::ios::binary);
    MY_ASSERT(samples_file.good(), "Failed to open samples.bin");
    const uint16_t samples_buflen = 1024;
    char samples_buffer[samples_buflen];
    uint32_t samples_offset = SAMPLES_OFFSET;
    while (!samples_file.eof()) {
        samples_file.read(samples_buffer, samples_buflen);
        int16_t read_len = samples_file.gcount();
        write_to_nvm(samples_buffer, samples_offset, read_len);
        samples_offset += read_len;
        my_printf_debug("Copied %d bytes of samples data" NEWLINE, read_len);
    }

    // others
    write_to_nvm_segmented(node_flags_data, NODE_FLAGS_OFFSET, NODE_FLAGS_DATA_LEN);
}

void notify_layer_finished(void) {}
void notify_model_finished(void) {}
void notify_indicator(uint8_t idx) {}
bool read_gpio_flag(GPIOFlag flag) { return false; }

[[ noreturn ]] void ERROR_OCCURRED(void) {
    exit_with_status(1);
}

#endif // PC_BUILD
