#include <driverlib.h>
#ifdef __MSP430__
#include <msp430.h>
#include <DSPLib.h>
#include "main.h"
#elif defined(__MSP432__)
#include <msp432.h>
#endif
#include <cstdint>
#include <cstring>
#include "intermittent-cnn.h"
#include "cnn_common.h"
#include "counters.h"
#include "platform.h"
#include "data.h"
#include "my_debug.h"
#include "tools/myuart.h"
#include "tools/our_misc.h"
#include "tools/dvfs.h"

#ifdef __MSP430__
#define DATA_SECTION_NVM _Pragma("DATA_SECTION(\".nvm\")")
#else
#define DATA_SECTION_NVM
#endif

#ifdef __MSP432__
uint32_t last_cyccnt = 0;
#endif

#ifdef __MSP430__

#define MY_DMA_CHANNEL DMA_CHANNEL_0

#endif

void my_memcpy(void* dest, const void* src, size_t n) {
#ifdef __MSP430__
    DMA0CTL = 0;

    DMACTL0 &= 0xFF00;
    // set DMA transfer trigger for channel 0
    DMACTL0 |= DMA0TSEL__DMAREQ;

    DMA_setSrcAddress(MY_DMA_CHANNEL, (uint32_t)src, DMA_DIRECTION_INCREMENT);
    DMA_setDstAddress(MY_DMA_CHANNEL, (uint32_t)dest, DMA_DIRECTION_INCREMENT);
    /* transfer size is in words (2 bytes) */
    DMA0SZ = n >> 1;
    DMA0CTL |= DMAEN + DMA_TRANSFER_BLOCK + DMA_SIZE_SRCWORD_DSTWORD;
    DMA0CTL |= DMAREQ;
#elif defined(__MSP432__)
    MAP_DMA_enableModule();
    MAP_DMA_setControlBase(controlTable);
    MAP_DMA_setChannelControl(
        DMA_CH0_RESERVED0 | UDMA_PRI_SELECT, // Channel 0, PRImary channel
        // re-arbitrate after 1024 (maximum) items
        // an item is 16-bit
        UDMA_ARB_1024 | UDMA_SIZE_16 | UDMA_SRC_INC_16 | UDMA_DST_INC_16
    );
    // Use the first configurable DMA interrupt handler DMA_INT1_IRQHandler,
    // which is defined below (overriding weak symbol in startup*.c)
    MAP_DMA_assignInterrupt(DMA_INT1, 0);
    MAP_Interrupt_enableInterrupt(INT_DMA_INT1);
    MAP_Interrupt_disableSleepOnIsrExit();
    MAP_DMA_setChannelTransfer(
        DMA_CH0_RESERVED0 | UDMA_PRI_SELECT,
        UDMA_MODE_AUTO, // Set as auto mode with no need to retrigger after each arbitration
        const_cast<void*>(src), dest,
        n >> 1 // transfer size in items
    );
    curDMATransmitChannelNum = 0;
    MAP_DMA_enableChannel(0);
    MAP_DMA_requestSoftwareTransfer(0);
    while (MAP_DMA_isChannelEnabled(0)) {}
#endif
}

void my_memcpy_from_parameters(void *dest, const ParameterInfo *param, uint32_t offset_in_bytes, size_t n) {
    MY_ASSERT(offset_in_bytes + n <= PARAMETERS_DATA_LEN);
#if ENABLE_COUNTERS && !ENABLE_DEMO_COUNTERS
    if (counters_enabled) {
        add_counter(offsetof(Counters, nvm_read_parameters), n);
        my_printf_debug("Recorded %lu bytes fetched from parameters, accumulated=%" PRIu32 NEWLINE, n, get_counter(offsetof(Counters, nvm_read_parameters)));
    }
#endif

#if !DISABLE_FEATURE_MAP_NVM_ACCESS
    read_from_nvm(dest, PARAMETERS_OFFSET + param->params_offset + offset_in_bytes, n);
#endif
}

void read_from_nvm(void* vm_buffer, uint32_t nvm_offset, size_t n) {
    SPI_ADDR addr;
    addr.L = nvm_offset;
    MY_ASSERT(n <= 1024);
    SPI_READ(&addr, reinterpret_cast<uint8_t*>(vm_buffer), n);
}

struct GPIOPin {
    uint16_t port;
    uint16_t pin;
};

static GPIOPin indicators[] = {
#ifdef __MSP430__
    { GPIO_PORT_P4, GPIO_PIN7 }, // used in notify_layer_finished()
    { GPIO_PORT_P1, GPIO_PIN5 }, // TODO: check if it works
#else
    { GPIO_PORT_P5, GPIO_PIN4 }, // used in notify_layer_finished()
    { GPIO_PORT_P4, GPIO_PIN7 },
#endif
};

static GPIOPin gpio_flags[] = {
#ifdef __MSP430__
#error "TODO"
#else
    { GPIO_PORT_P2, GPIO_PIN7 },
    { GPIO_PORT_P2, GPIO_PIN6 },
    { GPIO_PORT_P2, GPIO_PIN4 },
#endif
};

void write_to_nvm(const void* vm_buffer, uint32_t nvm_offset, size_t n, uint16_t timer_delay) {
    SPI_ADDR addr;
    addr.L = nvm_offset;
    check_nvm_write_address(nvm_offset, n);
    MY_ASSERT(n <= 1024);
    SPI_WRITE2(&addr, reinterpret_cast<const uint8_t*>(vm_buffer), n, timer_delay);
    if (!timer_delay) {
        SPI_WAIT_DMA();
    }
}

void my_erase() {
    eraseFRAM2(0x00);
}

void copy_data_to_nvm(void) {
    write_to_nvm_segmented(samples_data, SAMPLES_OFFSET, SAMPLES_DATA_LEN);
    write_to_nvm_segmented(parameters_data, PARAMETERS_OFFSET, PARAMETERS_DATA_LEN);
    write_to_nvm_segmented(node_flags_data, NODE_FLAGS_OFFSET, NODE_FLAGS_DATA_LEN);
}

[[ noreturn ]] void ERROR_OCCURRED(void) {
    while (1);
}

#ifdef __MSP430__
#define GPIO_COUNTER_PORT GPIO_PORT_P8
#define GPIO_COUNTER_PIN GPIO_PIN0
#define GPIO_RESET_PORT GPIO_PORT_P5
#define GPIO_RESET_PIN GPIO_PIN7
#else
#define GPIO_COUNTER_PORT GPIO_PORT_P5
#define GPIO_COUNTER_PIN GPIO_PIN5
#define GPIO_RESET_PORT GPIO_PORT_P2
#define GPIO_RESET_PIN GPIO_PIN5
#endif

#define STABLE_POWER_ITERATIONS 10

void IntermittentCNNTest() {
    GPIO_setAsOutputPin(GPIO_COUNTER_PORT, GPIO_COUNTER_PIN);
    GPIO_setOutputLowOnPin(GPIO_COUNTER_PORT, GPIO_COUNTER_PIN);
    for (size_t idx = 0; idx < sizeof(indicators) / sizeof(indicators[0]); idx++) {
        GPIO_setAsOutputPin(indicators[idx].port, indicators[idx].pin);
        GPIO_setOutputLowOnPin(indicators[idx].port, indicators[idx].pin);
    }
    GPIO_setAsInputPinWithPullUpResistor(GPIO_RESET_PORT, GPIO_RESET_PIN);
    for (size_t idx = 0; idx < sizeof(gpio_flags) / sizeof(gpio_flags[0]); idx++) {
        GPIO_setAsInputPinWithPullUpResistor(gpio_flags[idx].port, gpio_flags[idx].pin);
    }

    GPIO_setAsOutputPin( GPIO_PORT_P1, GPIO_PIN0 );
    GPIO_setOutputHighOnPin(GPIO_PORT_P1, GPIO_PIN0);

    // sleep to wait for external FRAM
    // 5ms / (1/f)
    our_delay_cycles(5E-3 * getFrequency(FreqLevel));

    initSPI();
    if (testSPI() != 0) {
        // external FRAM failed to initialize - reset
        volatile uint16_t counter = 1000;
        // waiting some time seems to increase the possibility
        // of a successful FRAM initialization on next boot
        while (counter--);
        WDTCTL = 0;
    }

    load_model_from_nvm();

#if ENABLE_COUNTERS
    load_counters();
#endif

    if (!GPIO_getInputPinValue(GPIO_RESET_PORT, GPIO_RESET_PIN)) {
        uartinit();

        // To get counters in NVM after intermittent tests
        print_all_counters();

        first_run();

        notify_model_finished();

        for (uint8_t idx = 0; idx < STABLE_POWER_ITERATIONS; idx++) {
            run_cnn_tests(1);
        }

        my_printf("Done testing run" NEWLINE);

        // For platforms where counters are recorded in VM (ex: MSP432)
        print_all_counters();

        // for exhaustive search, which uses run_counter as the index of configurations
        get_model()->run_counter = 0;
        commit_model();

        while (1);
    }

#if ENABLE_DEMO_COUNTERS
    uartinit();
#endif
    while (1) {
        run_cnn_tests(1);
    }
}

void button_pushed(uint16_t button1_status, uint16_t button2_status) {
    my_printf_debug("button1_status=%d button2_status=%d" NEWLINE, button1_status, button2_status);
}

static void gpio_pulse(uint8_t port, uint16_t pin) {
    // Trigger a short peak so that multiple inferences in long power cycles are correctly recorded
    GPIO_setOutputHighOnPin(port, pin);
    our_delay_cycles(5E-3 * getFrequency(FreqLevel));
    GPIO_setOutputLowOnPin(port, pin);
}

void notify_layer_finished(void) {
    notify_indicator(0);
}

void notify_model_finished(void) {
    my_printf("." NEWLINE);
    gpio_pulse(GPIO_COUNTER_PORT, GPIO_COUNTER_PIN);
}

void notify_indicator(uint8_t idx) {
    my_printf("I%d" NEWLINE, idx);
    gpio_pulse(indicators[idx].port, indicators[idx].pin);
}

bool read_gpio_flag(GPIOFlag flag) {
    uint8_t idx = static_cast<uint8_t>(flag);
    return !GPIO_getInputPinValue(gpio_flags[idx].port, gpio_flags[idx].pin);
}
