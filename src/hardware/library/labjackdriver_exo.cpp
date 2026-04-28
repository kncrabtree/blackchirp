#ifndef Q_OS_WIN

#include "labjackdriver.h"
#include <hardware/optional/ioboard/u3.h>
#include <hardware/library/labjacklibrary.h>
#include <data/loghandler.h>

namespace BC::Labjack {

struct DeviceHandle {
    enum class Kind { U3 /*, U6 */ } kind;
    HANDLE h;
    union {
        u3CalibrationInfo u3Cal;
        // u6CalibrationInfo u6Cal;   // when U6 support lands
    };
};

static void destroyHandle(DeviceHandle *dh)
{
    if (dh && dh->h)
        closeUSBConnection(dh->h);
    delete dh;
}

bool isAvailable()
{
    return LabjackLibrary::instance().isAvailable();
}

QString errorString()
{
    return LabjackLibrary::instance().errorString();
}

HandlePtr openU3(int serialOrLocalId)
{
    HANDLE h = openUSBConnection(serialOrLocalId);
    if (!h)
        return HandlePtr(nullptr, destroyHandle);

    auto *dh = new DeviceHandle;
    dh->kind = DeviceHandle::Kind::U3;
    dh->h    = h;

    if (getCalibrationInfo(h, &dh->u3Cal) < 0) {
        closeUSBConnection(h);
        delete dh;
        return HandlePtr(nullptr, destroyHandle);
    }

    return HandlePtr(dh, destroyHandle);
}

bool readAnalog(DeviceHandle *dh, int channel, double &out)
{
    if (!dh)
        return false;
    switch (dh->kind) {
    case DeviceHandle::Kind::U3: {
        long err = eAIN(dh->h, &dh->u3Cal, 1, nullptr, channel, 31,
                        &out, 0, 0, 0, 0, 0, 0);
        if (err) {
            bcError(u"eAIN returned error %1"_s.arg(err));
            return false;
        }
        return true;
    }
    }
    return false;
}

bool readDigital(DeviceHandle *dh, int channel, bool &out)
{
    if (!dh)
        return false;
    switch (dh->kind) {
    case DeviceHandle::Kind::U3: {
        long state = 0;
        long err = eDI(dh->h, 1, channel, &state);
        if (err) {
            bcError(u"eDI returned error %1"_s.arg(err));
            return false;
        }
        out = state != 0;
        return true;
    }
    }
    return false;
}

bool writeAnalog(DeviceHandle *dh, int channel, double voltage)
{
    if (!dh)
        return false;
    switch (dh->kind) {
    case DeviceHandle::Kind::U3: {
        long err = eDAC(dh->h, &dh->u3Cal, 1, channel, voltage, 0, 0, 0);
        if (err) {
            bcError(u"eDAC returned error %1"_s.arg(err));
            return false;
        }
        return true;
    }
    }
    return false;
}

bool writeDigital(DeviceHandle *dh, int channel, bool state)
{
    if (!dh)
        return false;
    switch (dh->kind) {
    case DeviceHandle::Kind::U3: {
        long err = eDO(dh->h, 1, channel, state ? 1L : 0L);
        if (err) {
            bcError(u"eDO returned error %1"_s.arg(err));
            return false;
        }
        return true;
    }
    }
    return false;
}

bool configureTimers(DeviceHandle *dh,
                     std::array<long,2>   enableTimers,
                     std::array<long,2>   enableCounters,
                     long                 pinOffset,
                     long                 timerClockBaseIdx,
                     long                 timerClockDivisor,
                     std::array<long,2>   timerModes,
                     std::array<double,2> timerValues)
{
    if (!dh)
        return false;
    long err = eTCConfig(dh->h,
                         enableTimers.data(), enableCounters.data(),
                         pinOffset, timerClockBaseIdx, timerClockDivisor,
                         timerModes.data(), timerValues.data(),
                         0, 0);
    if (err) {
        bcError(u"eTCConfig returned error %1"_s.arg(err));
        return false;
    }
    return true;
}

bool readTimers(DeviceHandle *dh,
                std::array<long,2>    readTimersArg,
                std::array<long,2>    updateResetTimers,
                std::array<long,2>    readCounters,
                std::array<long,2>    resetCounters,
                std::array<double,2> &timerValues,
                std::array<double,2> &counterValues)
{
    if (!dh)
        return false;
    long err = eTCValues(dh->h,
                         readTimersArg.data(), updateResetTimers.data(),
                         readCounters.data(), resetCounters.data(),
                         timerValues.data(), counterValues.data(),
                         0, 0);
    if (err) {
        bcError(u"eTCValues returned error %1"_s.arg(err));
        return false;
    }
    return true;
}

} // namespace BC::Labjack

#endif // !Q_OS_WIN
