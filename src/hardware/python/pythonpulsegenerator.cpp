#include "pythonpulsegenerator.h"

#include <QJsonObject>

#include <hardware/core/hardwareregistration.h>
#include <data/settings/hardwarekeys.h>
#include <data/bcglobals.h>

// ============================================================================
// Registration
// ============================================================================
REGISTER_HARDWARE_META(PythonPulseGenerator, "Python Pulse Generator (user-defined Python script)")
REGISTER_HARDWARE_PROTOCOLS(PythonPulseGenerator, CommunicationProtocol::Rs232, CommunicationProtocol::Tcp, CommunicationProtocol::Virtual)
REGISTER_HARDWARE_SETTINGS(PythonPulseGenerator,
    {BC::Key::PGen::numChannels, "Number of Channels",
     "Number of pulse output channels",
     8, 1, 64, HwSettingPriority::Required}
)

// ============================================================================
// Constructor
// ============================================================================
PythonPulseGenerator::PythonPulseGenerator(const QString &label, QObject *parent) :
    PulseGenerator(
        QString(PythonPulseGenerator::staticMetaObject.className()),
        label,
        [&label]() -> int {
            SettingsStorage s(BC::Key::hwKey(
                                  QString(PulseGenerator::staticMetaObject.className()),
                                  label));
            return s.get(BC::Key::PGen::numChannels, 8);
        }(),
        parent),
    PythonHardwareBase(d_key, d_model)
{
    d_threaded = true;
}

// ============================================================================
// initializePGen()
// ============================================================================
void PythonPulseGenerator::initializePGen()
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
// testConnection()
// ============================================================================
bool PythonPulseGenerator::testConnection()
{
    if (!testPythonConnection(p_comm)) {
        d_errorString = pythonErrorString();
        return false;
    }
    return true;
}

// ============================================================================
// readSettings()
// ============================================================================
void PythonPulseGenerator::readSettings()
{
    pythonReadSettings();
}

// ============================================================================
// Per-channel set virtuals
// ============================================================================

bool PythonPulseGenerator::setChWidth(const int index, const double width)
{
    if (!pu_process || !pu_process->isRunning())
        return false;

    QJsonObject req;
    req[QStringLiteral("method")]  = QStringLiteral("set_ch_width");
    req[QStringLiteral("channel")] = index;
    req[QStringLiteral("width")]   = width;
    auto resp = pu_process->sendRequest(req);
    return !resp.contains(QStringLiteral("error")) &&
           resp[QStringLiteral("result")].toBool(false);
}

bool PythonPulseGenerator::setChDelay(const int index, const double delay)
{
    if (!pu_process || !pu_process->isRunning())
        return false;

    QJsonObject req;
    req[QStringLiteral("method")]  = QStringLiteral("set_ch_delay");
    req[QStringLiteral("channel")] = index;
    req[QStringLiteral("delay")]   = delay;
    auto resp = pu_process->sendRequest(req);
    return !resp.contains(QStringLiteral("error")) &&
           resp[QStringLiteral("result")].toBool(false);
}

bool PythonPulseGenerator::setChActiveLevel(const int index, const PulseGenConfig::ActiveLevel level)
{
    if (!pu_process || !pu_process->isRunning())
        return false;

    QJsonObject req;
    req[QStringLiteral("method")]  = QStringLiteral("set_ch_active_level");
    req[QStringLiteral("channel")] = index;
    req[QStringLiteral("level")]   = static_cast<int>(level);
    auto resp = pu_process->sendRequest(req);
    return !resp.contains(QStringLiteral("error")) &&
           resp[QStringLiteral("result")].toBool(false);
}

bool PythonPulseGenerator::setChEnabled(const int index, const bool en)
{
    if (!pu_process || !pu_process->isRunning())
        return false;

    QJsonObject req;
    req[QStringLiteral("method")]   = QStringLiteral("set_ch_enabled");
    req[QStringLiteral("channel")]  = index;
    req[QStringLiteral("enabled")]  = en;
    auto resp = pu_process->sendRequest(req);
    return !resp.contains(QStringLiteral("error")) &&
           resp[QStringLiteral("result")].toBool(false);
}

bool PythonPulseGenerator::setChSyncCh(const int index, const int syncCh)
{
    if (!pu_process || !pu_process->isRunning())
        return false;

    QJsonObject req;
    req[QStringLiteral("method")]   = QStringLiteral("set_ch_sync_ch");
    req[QStringLiteral("channel")]  = index;
    req[QStringLiteral("sync_ch")]  = syncCh;
    auto resp = pu_process->sendRequest(req);
    return !resp.contains(QStringLiteral("error")) &&
           resp[QStringLiteral("result")].toBool(false);
}

bool PythonPulseGenerator::setChMode(const int index, const PulseGenConfig::ChannelMode mode)
{
    if (!pu_process || !pu_process->isRunning())
        return false;

    QJsonObject req;
    req[QStringLiteral("method")]  = QStringLiteral("set_ch_mode");
    req[QStringLiteral("channel")] = index;
    req[QStringLiteral("mode")]    = static_cast<int>(mode);
    auto resp = pu_process->sendRequest(req);
    return !resp.contains(QStringLiteral("error")) &&
           resp[QStringLiteral("result")].toBool(false);
}

bool PythonPulseGenerator::setChDutyOn(const int index, const int pulses)
{
    if (!pu_process || !pu_process->isRunning())
        return false;

    QJsonObject req;
    req[QStringLiteral("method")]  = QStringLiteral("set_ch_duty_on");
    req[QStringLiteral("channel")] = index;
    req[QStringLiteral("pulses")]  = pulses;
    auto resp = pu_process->sendRequest(req);
    return !resp.contains(QStringLiteral("error")) &&
           resp[QStringLiteral("result")].toBool(false);
}

bool PythonPulseGenerator::setChDutyOff(const int index, const int pulses)
{
    if (!pu_process || !pu_process->isRunning())
        return false;

    QJsonObject req;
    req[QStringLiteral("method")]  = QStringLiteral("set_ch_duty_off");
    req[QStringLiteral("channel")] = index;
    req[QStringLiteral("pulses")]  = pulses;
    auto resp = pu_process->sendRequest(req);
    return !resp.contains(QStringLiteral("error")) &&
           resp[QStringLiteral("result")].toBool(false);
}

// ============================================================================
// Global set virtuals
// ============================================================================

bool PythonPulseGenerator::setHwPulseMode(PulseGenConfig::PGenMode mode)
{
    if (!pu_process || !pu_process->isRunning())
        return false;

    QJsonObject req;
    req[QStringLiteral("method")] = QStringLiteral("set_hw_pulse_mode");
    req[QStringLiteral("mode")]   = static_cast<int>(mode);
    auto resp = pu_process->sendRequest(req);
    return !resp.contains(QStringLiteral("error")) &&
           resp[QStringLiteral("result")].toBool(false);
}

bool PythonPulseGenerator::setHwRepRate(double rr)
{
    if (!pu_process || !pu_process->isRunning())
        return false;

    QJsonObject req;
    req[QStringLiteral("method")]   = QStringLiteral("set_hw_rep_rate");
    req[QStringLiteral("rep_rate")] = rr;
    auto resp = pu_process->sendRequest(req);
    return !resp.contains(QStringLiteral("error")) &&
           resp[QStringLiteral("result")].toBool(false);
}

bool PythonPulseGenerator::setHwPulseEnabled(bool en)
{
    if (!pu_process || !pu_process->isRunning())
        return false;

    QJsonObject req;
    req[QStringLiteral("method")]  = QStringLiteral("set_hw_pulse_enabled");
    req[QStringLiteral("enabled")] = en;
    auto resp = pu_process->sendRequest(req);
    return !resp.contains(QStringLiteral("error")) &&
           resp[QStringLiteral("result")].toBool(false);
}

// ============================================================================
// Per-channel read virtuals
// ============================================================================

double PythonPulseGenerator::readChWidth(const int index)
{
    if (!pu_process || !pu_process->isRunning())
        return 0.0;

    QJsonObject req;
    req[QStringLiteral("method")]  = QStringLiteral("read_ch_width");
    req[QStringLiteral("channel")] = index;
    auto resp = pu_process->sendRequest(req);

    if (resp.contains(QStringLiteral("error")))
        return 0.0;

    return resp[QStringLiteral("result")].toDouble(0.0);
}

double PythonPulseGenerator::readChDelay(const int index)
{
    if (!pu_process || !pu_process->isRunning())
        return 0.0;

    QJsonObject req;
    req[QStringLiteral("method")]  = QStringLiteral("read_ch_delay");
    req[QStringLiteral("channel")] = index;
    auto resp = pu_process->sendRequest(req);

    if (resp.contains(QStringLiteral("error")))
        return 0.0;

    return resp[QStringLiteral("result")].toDouble(0.0);
}

PulseGenConfig::ActiveLevel PythonPulseGenerator::readChActiveLevel(const int index)
{
    if (!pu_process || !pu_process->isRunning())
        return PulseGenConfig::ActiveHigh;

    QJsonObject req;
    req[QStringLiteral("method")]  = QStringLiteral("read_ch_active_level");
    req[QStringLiteral("channel")] = index;
    auto resp = pu_process->sendRequest(req);

    if (resp.contains(QStringLiteral("error")))
        return PulseGenConfig::ActiveHigh;

    return static_cast<PulseGenConfig::ActiveLevel>(
        resp[QStringLiteral("result")].toInt(
            static_cast<int>(PulseGenConfig::ActiveHigh)));
}

bool PythonPulseGenerator::readChEnabled(const int index)
{
    if (!pu_process || !pu_process->isRunning())
        return false;

    QJsonObject req;
    req[QStringLiteral("method")]  = QStringLiteral("read_ch_enabled");
    req[QStringLiteral("channel")] = index;
    auto resp = pu_process->sendRequest(req);

    if (resp.contains(QStringLiteral("error")))
        return false;

    return resp[QStringLiteral("result")].toBool(false);
}

int PythonPulseGenerator::readChSynchCh(const int index)
{
    if (!pu_process || !pu_process->isRunning())
        return 0;

    QJsonObject req;
    req[QStringLiteral("method")]  = QStringLiteral("read_ch_sync_ch");
    req[QStringLiteral("channel")] = index;
    auto resp = pu_process->sendRequest(req);

    if (resp.contains(QStringLiteral("error")))
        return 0;

    return resp[QStringLiteral("result")].toInt(0);
}

PulseGenConfig::ChannelMode PythonPulseGenerator::readChMode(const int index)
{
    if (!pu_process || !pu_process->isRunning())
        return PulseGenConfig::Normal;

    QJsonObject req;
    req[QStringLiteral("method")]  = QStringLiteral("read_ch_mode");
    req[QStringLiteral("channel")] = index;
    auto resp = pu_process->sendRequest(req);

    if (resp.contains(QStringLiteral("error")))
        return PulseGenConfig::Normal;

    return static_cast<PulseGenConfig::ChannelMode>(
        resp[QStringLiteral("result")].toInt(
            static_cast<int>(PulseGenConfig::Normal)));
}

int PythonPulseGenerator::readChDutyOn(const int index)
{
    if (!pu_process || !pu_process->isRunning())
        return 1;

    QJsonObject req;
    req[QStringLiteral("method")]  = QStringLiteral("read_ch_duty_on");
    req[QStringLiteral("channel")] = index;
    auto resp = pu_process->sendRequest(req);

    if (resp.contains(QStringLiteral("error")))
        return 1;

    return resp[QStringLiteral("result")].toInt(1);
}

int PythonPulseGenerator::readChDutyOff(const int index)
{
    if (!pu_process || !pu_process->isRunning())
        return 1;

    QJsonObject req;
    req[QStringLiteral("method")]  = QStringLiteral("read_ch_duty_off");
    req[QStringLiteral("channel")] = index;
    auto resp = pu_process->sendRequest(req);

    if (resp.contains(QStringLiteral("error")))
        return 1;

    return resp[QStringLiteral("result")].toInt(1);
}

// ============================================================================
// Global read virtuals
// ============================================================================

PulseGenConfig::PGenMode PythonPulseGenerator::readHwPulseMode()
{
    if (!pu_process || !pu_process->isRunning())
        return PulseGenConfig::Continuous;

    QJsonObject req;
    req[QStringLiteral("method")] = QStringLiteral("read_hw_pulse_mode");
    auto resp = pu_process->sendRequest(req);

    if (resp.contains(QStringLiteral("error")))
        return PulseGenConfig::Continuous;

    return static_cast<PulseGenConfig::PGenMode>(
        resp[QStringLiteral("result")].toInt(
            static_cast<int>(PulseGenConfig::Continuous)));
}

double PythonPulseGenerator::readHwRepRate()
{
    if (!pu_process || !pu_process->isRunning())
        return 0.0;

    QJsonObject req;
    req[QStringLiteral("method")] = QStringLiteral("read_hw_rep_rate");
    auto resp = pu_process->sendRequest(req);

    if (resp.contains(QStringLiteral("error")))
        return 0.0;

    return resp[QStringLiteral("result")].toDouble(0.0);
}

bool PythonPulseGenerator::readHwPulseEnabled()
{
    if (!pu_process || !pu_process->isRunning())
        return false;

    QJsonObject req;
    req[QStringLiteral("method")] = QStringLiteral("read_hw_pulse_enabled");
    auto resp = pu_process->sendRequest(req);

    if (resp.contains(QStringLiteral("error")))
        return false;

    return resp[QStringLiteral("result")].toBool(false);
}
