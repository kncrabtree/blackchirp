#include "pythonclock.h"

#include <QJsonObject>

#include <hardware/core/hardwareregistration.h>

// ============================================================================
// Registration
// ============================================================================
REGISTER_HARDWARE_META(PythonClock, "Python Clock (user-defined Python script)")
REGISTER_HARDWARE_PROTOCOLS(PythonClock, CommunicationProtocol::Rs232, CommunicationProtocol::Tcp, CommunicationProtocol::Virtual)
REGISTER_HARDWARE_PARAMS(PythonClock)

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

    setDefault(BC::Key::Clock::minFreq, 0.0);
    setDefault(BC::Key::Clock::maxFreq, 1e7);

    save();
}

// ============================================================================
// configParams()
// ============================================================================
QVector<HwConfigParam> PythonClock::configParams()
{
    return {
        { BC::Key::PythonClock::numOutputs, QStringLiteral("Number of Outputs"), QVariant(1), QVariant(1), QVariant(8) },
        { BC::Key::Clock::tunable,          QStringLiteral("Tunable"),           QVariant(true), 0, 0 },
    };
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

    connect(pu_process.get(), &PythonProcess::logMessage,
            this, &PythonClock::logMessage);
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
    req[QStringLiteral("method")] = QStringLiteral("hw_set_frequency");
    req[QStringLiteral("freq_mhz")] = freqMHz;
    req[QStringLiteral("output")] = outputIndex;
    auto resp = pu_process->sendRequest(req);

    if (resp.contains(QStringLiteral("error")))
        return false;

    return resp[QStringLiteral("result")].toBool(false);
}

// ============================================================================
// readHwFrequency()
// ============================================================================
double PythonClock::readHwFrequency(int outputIndex)
{
    if (!pu_process || !pu_process->isRunning())
        return -1.0;

    QJsonObject req;
    req[QStringLiteral("method")] = QStringLiteral("hw_read_frequency");
    req[QStringLiteral("output")] = outputIndex;
    auto resp = pu_process->sendRequest(req);

    if (resp.contains(QStringLiteral("error")))
        return -1.0;

    return resp[QStringLiteral("result")].toDouble(-1.0);
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

// ============================================================================
// forbiddenKeys()
// ============================================================================
QStringList PythonClock::forbiddenKeys() const
{
    auto keys = pythonForbiddenKeys();
    keys << BC::Key::PythonClock::numOutputs << BC::Key::Clock::tunable;
    return keys;
}
