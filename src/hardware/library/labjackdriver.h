#ifndef LABJACKDRIVER_H
#define LABJACKDRIVER_H

#include <array>
#include <memory>

#include <QString>

namespace BC::Labjack {

struct DeviceHandle;
using HandlePtr = std::unique_ptr<DeviceHandle, void(*)(DeviceHandle*)>;

namespace Const {
    // Timer clocks (U3 hardware version 1.21+)
    inline constexpr long tc48MHZ     = 22;
    inline constexpr long tc48MHZ_DIV = 26;
    inline constexpr long tc12MHZ     = 21;
    inline constexpr long tc12MHZ_DIV = 25;
    inline constexpr long tc4MHZ      = 20;
    inline constexpr long tc4MHZ_DIV  = 24;
    inline constexpr long tc1MHZ_DIV  = 23;
    // Timer clocks (U3 hardware version 1.20 and lower)
    inline constexpr long tc24MHZ     = 12;
    inline constexpr long tc6MHZ      = 11;
    inline constexpr long tc2MHZ      = 10;

    // UD device type / connection type (used by Windows backend)
    inline constexpr long dtU3  = 3;
    inline constexpr long ctUSB = 1;
} // namespace Const

bool    isAvailable();
QString errorString();

// Returns null HandlePtr on failure.
HandlePtr openU3(int serialOrLocalId);
// HandlePtr openU6(int serialOrLocalId);  // added when U6 support lands

bool readAnalog  (DeviceHandle*, int channel, double &out);
bool readDigital (DeviceHandle*, int channel, bool   &out);
bool writeAnalog (DeviceHandle*, int channel, double  voltage);
bool writeDigital(DeviceHandle*, int channel, bool    state);

bool configureTimers(DeviceHandle*,
                     std::array<long,2>   enableTimers,
                     std::array<long,2>   enableCounters,
                     long                 pinOffset,
                     long                 timerClockBaseIdx,
                     long                 timerClockDivisor,
                     std::array<long,2>   timerModes,
                     std::array<double,2> timerValues);

bool readTimers(DeviceHandle*,
                std::array<long,2>    readTimers,
                std::array<long,2>    updateResetTimers,
                std::array<long,2>    readCounters,
                std::array<long,2>    resetCounters,
                std::array<double,2> &timerValues,
                std::array<double,2> &counterValues);

} // namespace BC::Labjack

#endif // LABJACKDRIVER_H
