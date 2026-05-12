#include "labjackdriver.h"
#include <hardware/library/labjacklibrary.h>
#include <data/loghandler.h>

namespace BC::Labjack {

struct DeviceHandle {
    enum class Kind { U3 /*, U6 */ } kind;
    long h;  // LJ_HANDLE
};

static void destroyHandle(DeviceHandle *dh)
{
    if (!dh)
        return;
    auto &lib = LabjackLibrary::instance();
    if (lib.Close && dh->h)
        lib.Close(dh->h);
    delete dh;
}

static void logUDError(long err)
{
    auto &lib = LabjackLibrary::instance();
    if (lib.ErrorToString) {
        char buf[256] = {};
        lib.ErrorToString(err, buf);
        bcError(QString::fromLatin1(buf));
    } else {
        bcError(u"LabJack UD error %1"_s.arg(err));
    }
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
    auto &lib = LabjackLibrary::instance();
    if (!lib.OpenLabJack)
        return HandlePtr(nullptr, destroyHandle);

    long firstFound = (serialOrLocalId < 0) ? 1L : 0L;
    QByteArray addr = (serialOrLocalId < 0)
                      ? QByteArray("1")
                      : QByteArray::number(serialOrLocalId);

    long h = 0;
    long err = lib.OpenLabJack(Const::dtU3, Const::ctUSB,
                               addr.constData(), firstFound, &h);
    if (err) {
        logUDError(err);
        return HandlePtr(nullptr, destroyHandle);
    }

    auto *dh = new DeviceHandle;
    dh->kind = DeviceHandle::Kind::U3;
    dh->h    = h;
    return HandlePtr(dh, destroyHandle);
}

bool readAnalog(DeviceHandle *dh, int channel, double &out)
{
    if (!dh)
        return false;
    auto &lib = LabjackLibrary::instance();
    long err = lib.eAIN(dh->h, static_cast<long>(channel), 31L,
                        &out, 0L, 0L, 0L, 0L, 0L, 0L, 0.0, 0.0);
    if (err) {
        logUDError(err);
        return false;
    }
    return true;
}

bool readDigital(DeviceHandle *dh, int channel, bool &out)
{
    if (!dh)
        return false;
    auto &lib = LabjackLibrary::instance();
    long state = 0;
    long err = lib.eDI(dh->h, static_cast<long>(channel), &state);
    if (err) {
        logUDError(err);
        return false;
    }
    out = state != 0;
    return true;
}

bool writeAnalog(DeviceHandle *dh, int channel, double voltage)
{
    if (!dh)
        return false;
    auto &lib = LabjackLibrary::instance();
    long err = lib.eDAC(dh->h, static_cast<long>(channel), voltage,
                        0L, 0L, 0.0);
    if (err) {
        logUDError(err);
        return false;
    }
    return true;
}

bool writeDigital(DeviceHandle *dh, int channel, bool state)
{
    if (!dh)
        return false;
    auto &lib = LabjackLibrary::instance();
    long err = lib.eDO(dh->h, static_cast<long>(channel), state ? 1L : 0L);
    if (err) {
        logUDError(err);
        return false;
    }
    return true;
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
    auto &lib = LabjackLibrary::instance();
    long err = lib.eTCConfig(dh->h,
                             enableTimers.data(), enableCounters.data(),
                             pinOffset, timerClockBaseIdx, timerClockDivisor,
                             timerModes.data(), timerValues.data(),
                             0L, 0.0);
    if (err) {
        logUDError(err);
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
    auto &lib = LabjackLibrary::instance();
    long err = lib.eTCValues(dh->h,
                             readTimersArg.data(), updateResetTimers.data(),
                             readCounters.data(), resetCounters.data(),
                             timerValues.data(), counterValues.data(),
                             0L, 0.0);
    if (err) {
        logUDError(err);
        return false;
    }
    return true;
}

} // namespace BC::Labjack
