/*
 * firFilter.h
 *
 *  Created on: 2018�~3��19��
 *      Author: Meenchen
 */


/*******************************************************************************
*
* Name : FIR Filter
* Purpose : Benchmark an FIR filter. The input values for the filter
* is an array of 51 16-bit values. The order of the filter is
* 17.
*
*******************************************************************************/
#ifdef MSP430
#include "msp430x14x.h"
#endif
#include <math.h>
#include <config.h>

#define TaskID 5
extern unsigned long timeCounter;
extern unsigned long taskRecency[12];
extern unsigned long information[10];

#define FIR_LENGTH 17
#pragma NOINIT(FCOEFF)
const float FCOEFF[FIR_LENGTH] =
{
    -0.000091552734, 0.000305175781, 0.004608154297, 0.003356933594, -0.025939941406,
    -0.044006347656, 0.063079833984, 0.290313720703, 0.416748046875, 0.290313720703,
    0.063079833984, -0.044006347656, -0.025939941406, 0.003356933594, 0.004608154297,
    0.000305175781, -0.000091552734};

float SCOEFF[FIR_LENGTH];

/* The following array simulates input A/D converted values */
#pragma NOINIT(FINPUT)
const unsigned int FINPUT[] =
{
    0x0000, 0x0000, 0x0000, 0x0000,0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000,0x0000, 0x0000, 0x0000, 0x0000,
    0x0400, 0x0800, 0x0C00, 0x1000, 0x1400, 0x1800, 0x1C00, 0x2000,
    0x2400, 0x2000, 0x1C00, 0x1800, 0x1400, 0x1000, 0x0C00, 0x0800,
    0x0400, 0x0400, 0x0800, 0x0C00, 0x1000, 0x1400, 0x1800, 0x1C00,
    0x2000, 0x2400, 0x2000, 0x1C00, 0x1800, 0x1400, 0x1000, 0x0C00,
    0x0800, 0x0400, 0x0400, 0x0800, 0x0C00, 0x1000, 0x1400, 0x1800,
    0x1C00, 0x2000, 0x2400, 0x2000, 0x1C00, 0x1800, 0x1400, 0x1000,
    0x0C00, 0x0800, 0x0400};

unsigned int SINPUT[67];

#ifdef ONVM
void readback_fir()
{
    int i;
    for(i = 0; i < 67; i++)
        SINPUT[i] = FINPUT[i];
    for(i = 0; i < FIR_LENGTH; i++)
        SCOEFF[i] = FCOEFF[i];
}
#endif

#pragma NOINIT(Fsum)
volatile float Fsum;
volatile float Ssum;

void firFilter()
{
    int i, y; /* Loop counters */
    volatile float OUTPUT[36];
    int k;

#ifdef ONVM
    readback_fir();
#endif

    while(1){
        for(k = 0; k < ITERFIR; k++){
            for(y = 0; y < 36; y++)
            {
                Fsum=0;
                Ssum=0;
                for(i = 0; i < FIR_LENGTH/2; i++)
                {
#ifdef ONNVM
                    Fsum = Fsum+FCOEFF[i] * ( FINPUT[y + 16 - i] + FINPUT[y + i] );
                    OUTPUT[y] = Fsum;
#endif
#ifdef OUR
                    Ssum = Ssum+FCOEFF[i] * ( FINPUT[y + 16 - i] + FINPUT[y + i] );
                    OUTPUT[y] = Ssum;
#endif
#ifdef ONEVERSION
                    Ssum = Ssum+FCOEFF[i] * ( FINPUT[y + 16 - i] + FINPUT[y + i] );
                    OUTPUT[y] = Ssum;
#endif
#ifdef ONVM
                    Ssum = Ssum+SCOEFF[i] * ( SINPUT[y + 16 - i] + SINPUT[y + i] );
                    OUTPUT[y] = Ssum;
#endif
                }
            }
        }
        information[IDFIR]++;
    }
}
