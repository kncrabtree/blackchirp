#include "pythonclock.h"

#include <QJsonObject>

#include <hardware/core/hardwareregistration.h>

// ============================================================================
// Registration
// ============================================================================
REGISTER_HARDWARE_META(PythonClock, "Python Clock (user-defined Python script)")
REGISTER_HARDWARE_PROTOCOLS(PythonClock, CommunicationProtocol::Rs232, CommunicationProtocol::Tcp, CommunicationProtocol::Virtual)
REGISTER_HARDWARE_SETTINGS(PythonClock,
    {BC::Key::PythonClock::numOutputs, "Number of Outputs", "Number of independent frequency outputs on this clock.", 1, 1, 8, HwSettingPriority::Required},
    {BC::Key::Clock::tunable, "Tunable", "Clock frequency can be changed at runtime.", true, QVariant{}, QVariant{}, HwSettingPriority::Required}
)

// ============================================================================
// Constructor / Destructor
// ============================================================================
PythonClock::PythonClock(const QString &label, QObject *parent) :
    Clock(
        [&label]() -> int {
            SettingsStorage s(BC::Key::hwKey(
                                  QString(Clock::staticMetaObject.className()),
                                  label)
                              );
            auto n = s.get(BC::Key::PythonClock::numOutputs, 1);
            return n;
        }(),
        [&label]() -> bool {
            SettingsStorage s(BC::Key::hwKey(
                                  QString(Clock::staticMetaObject.className()),
                                  label)
                              );
            auto t = s.get(BC::Key::Clock::tunable, true);
            return t;
        }(),
        QString(PythonClock::staticMetaObject.className()),
        label,
        parent),
    PythonHardwareBase(d_key, d_model)
{
    d_threaded = true;

    save();
}


// ============================================================================
// initializeClock()
// ============================================================================
void PythonClock::initializeClock()
{
    initPythonProcess(p_comm,
        [this](const QString &key, const QVariant &defaultVal) -> QVariant {
            return get(key, defaultVal);
        },
        [this](const QString &key, const QVariant &val) {
            set(key, val, true);
        }
    );

}

// ============================================================================
// testClockConnection()
// ============================================================================
bool PythonClock::testClockConnection()
{
    if (!testPythonConnection(p_comm)) {
        d_errorString = pythonErrorString();
        return false;
    }
    return true;
}

// ============================================================================
// setHwFrequency()
// ============================================================================
bool PythonClock::setHwFrequency(double freqMHz, int outputIndex)
{
    if (!pu_process || !pu_process->isRunning())
        return false;

    QJsonObject req;
    req["method"_L1] = "hw_set_frequency"_L1;
    req["freq_mhz"_L1] = freqMHz;
    req["output"_L1] = outputIndex;
    auto resp = pu_process->sendRequest(req);

    if (resp.contains("error"_L1))
        return false;

    return resp["result"_L1].toBool(false);
}

// ============================================================================
// readHwFrequency()
// ============================================================================
double PythonClock::readHwFrequency(int outputIndex)
{
    if (!pu_process || !pu_process->isRunning())
        return -1.0;

    QJsonObject req;
    req["method"_L1] = "hw_read_frequency"_L1;
    req["output"_L1] = outputIndex;
    auto resp = pu_process->sendRequest(req);

    if (resp.contains("error"_L1))
        return -1.0;

    return resp["result"_L1].toDouble(-1.0);
}

// ============================================================================
// readSettings()
// ============================================================================
void PythonClock::readSettings()
{
    pythonReadSettings();
}

// ============================================================================
// sleep()
// ============================================================================
void PythonClock::sleep(bool b)
{
    pythonSleep(b);
}

