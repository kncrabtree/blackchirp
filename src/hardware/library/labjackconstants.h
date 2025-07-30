#ifndef LABJACKCONSTANTS_H
#define LABJACKCONSTANTS_H

/*!
 * \brief LabJack U3 constants without requiring vendor headers
 * 
 * This file defines essential constants from the LabJack U3 SDK without
 * requiring the vendor headers to be present at compile time. These constants
 * are used by BlackChirp hardware implementations that use dynamic LabJack
 * library loading.
 * 
 * Constants are organized under the LabJack::U3 namespace for future
 * extensibility with other LabJack device families (U6, T7, etc.).
 */

namespace LabJack::U3 {

// Timer clocks for Hardware Version 1.21 or higher
constexpr long LJ_tc4MHZ = 20;          /*!< 4 MHz timer clock */
constexpr long LJ_tc12MHZ = 21;         /*!< 12 MHz timer clock */
constexpr long LJ_tc48MHZ = 22;         /*!< 48 MHz timer clock */
constexpr long LJ_tc1MHZ_DIV = 23;      /*!< 1/Divisor MHz timer clock */
constexpr long LJ_tc4MHZ_DIV = 24;      /*!< 4/Divisor MHz timer clock */
constexpr long LJ_tc12MHZ_DIV = 25;     /*!< 12/Divisor MHz timer clock */
constexpr long LJ_tc48MHZ_DIV = 26;     /*!< 48/Divisor MHz timer clock */

// Timer clocks for Hardware Version 1.20 or lower
constexpr long LJ_tc2MHZ = 10;          /*!< 2 MHz timer clock */
constexpr long LJ_tc6MHZ = 11;          /*!< 6 MHz timer clock */
constexpr long LJ_tc24MHZ = 12;         /*!< 24 MHz timer clock */
constexpr long LJ_tc500KHZ_DIV = 13;    /*!< 500/Divisor KHz timer clock */
constexpr long LJ_tc2MHZ_DIV = 14;      /*!< 2/Divisor MHz timer clock */
constexpr long LJ_tc6MHZ_DIV = 15;      /*!< 6/Divisor MHz timer clock */
constexpr long LJ_tc24MHZ_DIV = 16;     /*!< 24/Divisor MHz timer clock */

// Timer modes
constexpr long LJ_tmPWM16 = 0;                  /*!< 16 bit PWM */
constexpr long LJ_tmPWM8 = 1;                   /*!< 8 bit PWM */
constexpr long LJ_tmRISINGEDGES32 = 2;          /*!< 32-bit rising to rising edge measurement */
constexpr long LJ_tmFALLINGEDGES32 = 3;         /*!< 32-bit falling to falling edge measurement */
constexpr long LJ_tmDUTYCYCLE = 4;              /*!< duty cycle measurement */
constexpr long LJ_tmFIRMCOUNTER = 5;            /*!< firmware based rising edge counter */
constexpr long LJ_tmFIRMCOUNTERDEBOUNCE = 6;    /*!< firmware counter with debounce */
constexpr long LJ_tmFREQOUT = 7;                /*!< frequency output */
constexpr long LJ_tmQUAD = 8;                   /*!< Quadrature */
constexpr long LJ_tmTIMERSTOP = 9;              /*!< stops another timer after n pulses */
constexpr long LJ_tmSYSTIMERLOW = 10;           /*!< read lower 32-bits of system timer */
constexpr long LJ_tmSYSTIMERHIGH = 11;          /*!< read upper 32-bits of system timer */
constexpr long LJ_tmRISINGEDGES16 = 12;         /*!< 16-bit rising to rising edge measurement */
constexpr long LJ_tmFALLINGEDGES16 = 13;        /*!< 16-bit falling to falling edge measurement */

} // namespace LabJack::U3

#endif // LABJACKCONSTANTS_H