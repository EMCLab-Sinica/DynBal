#include <driverlib.h>
#include "msp.h"
#include "plat-msp430.h"
#include "Tools/myuart.h"
#include "Tools/dvfs.h"

/**
 * main.c
 */

static void prvSetupHardware( void );
static void timerinit(void);

void main(void)
{
	WDT_A->CTL = WDT_A_CTL_PW | WDT_A_CTL_HOLD;		// stop watchdog timer

    setFrequency(FreqLevel);
    // XXX: disabled - timer intterupts appear to interfere DMA read for external FRAM
    // timerinit();

    prvSetupHardware();

    IntermittentCNNTest();
}

// See timer_a_upmode_gpio_toggle.c in MSP432 examples for code below

#define TIMER_PERIOD    375

static void prvSetupHardware( void ) {
    // Ref: MSP432 example gpio_input_interrupt.c

    /* Configuring P1.1 as an input and enabling interrupts */
    MAP_GPIO_setAsInputPinWithPullUpResistor(GPIO_PORT_P1, GPIO_PIN1|GPIO_PIN4);
    MAP_GPIO_clearInterruptFlag(GPIO_PORT_P1, GPIO_PIN1|GPIO_PIN4);
    MAP_GPIO_enableInterrupt(GPIO_PORT_P1, GPIO_PIN1|GPIO_PIN4);
    MAP_Interrupt_enableInterrupt(INT_PORT1);

    /* Enabling MASTER interrupts */
    MAP_Interrupt_enableMaster();
}

/* GPIO ISR */
void PORT1_IRQHandler(void)
{
    uint32_t status;

    status = MAP_GPIO_getEnabledInterruptStatus(GPIO_PORT_P1);
    MAP_GPIO_clearInterruptFlag(GPIO_PORT_P1, status);

    button_pushed(status & GPIO_PIN1, status & GPIO_PIN4);
}

/* Timer_A UpMode Configuration Parameter */
static const Timer_A_UpModeConfig upConfig = {
    .clockSource = TIMER_A_CLOCKSOURCE_SMCLK,
    .clockSourceDivider = TIMER_A_CLOCKSOURCE_DIVIDER_32,
    // SMCLK is 12MHz (from CS_getSMCLK()), so 1ms has 12M / 32 / 1000 = 375 ticks
    .timerPeriod = TIMER_PERIOD,
    .timerInterruptEnable_TAIE = TIMER_A_TAIE_INTERRUPT_DISABLE,
    .captureCompareInterruptEnable_CCR0_CCIE = TIMER_A_CCIE_CCR0_INTERRUPT_ENABLE,
    .timerClear = TIMER_A_DO_CLEAR
};

void timerinit(void) {
    /* Configuring Timer_A1 for Up Mode */
    MAP_Timer_A_configureUpMode(TIMER_A1_BASE, &upConfig);

    /* Enabling interrupts and starting the timer */
    MAP_Interrupt_enableInterrupt(INT_TA1_0);
    MAP_Timer_A_startCounter(TIMER_A1_BASE, TIMER_A_UP_MODE);

    /* Enabling MASTER interrupts */
    MAP_Interrupt_enableMaster();
}
