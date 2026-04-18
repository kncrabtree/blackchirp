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
    req["method"_L1]  = "set_ch_width"_L1;
    req["channel"_L1] = index;
    req["width"_L1]   = width;
    auto resp = pu_process->sendRequest(req);
    return !resp.contains("error"_L1) &&
           resp["result"_L1].toBool(false);
}

bool PythonPulseGenerator::setChDelay(const int index, const double delay)
{
    if (!pu_process || !pu_process->isRunning())
        return false;

    QJsonObject req;
    req["method"_L1]  = "set_ch_delay"_L1;
    req["channel"_L1] = index;
    req["delay"_L1]   = delay;
    auto resp = pu_process->sendRequest(req);
    return !resp.contains("error"_L1) &&
           resp["result"_L1].toBool(false);
}

bool PythonPulseGenerator::setChActiveLevel(const int index, const PulseGenConfig::ActiveLevel level)
{
    if (!pu_process || !pu_process->isRunning())
        return false;

    QJsonObject req;
    req["method"_L1]  = "set_ch_active_level"_L1;
    req["channel"_L1] = index;
    req["level"_L1]   = static_cast<int>(level);
    auto resp = pu_process->sendRequest(req);
    return !resp.contains("error"_L1) &&
           resp["result"_L1].toBool(false);
}

bool PythonPulseGenerator::setChEnabled(const int index, const bool en)
{
    if (!pu_process || !pu_process->isRunning())
        return false;

    QJsonObject req;
    req["method"_L1]   = "set_ch_enabled"_L1;
    req["channel"_L1]  = index;
    req["enabled"_L1]  = en;
    auto resp = pu_process->sendRequest(req);
    return !resp.contains("error"_L1) &&
           resp["result"_L1].toBool(false);
}

bool PythonPulseGenerator::setChSyncCh(const int index, const int syncCh)
{
    if (!pu_process || !pu_process->isRunning())
        return false;

    QJsonObject req;
    req["method"_L1]   = "set_ch_sync_ch"_L1;
    req["channel"_L1]  = index;
    req["sync_ch"_L1]  = syncCh;
    auto resp = pu_process->sendRequest(req);
    return !resp.contains("error"_L1) &&
           resp["result"_L1].toBool(false);
}

bool PythonPulseGenerator::setChMode(const int index, const PulseGenConfig::ChannelMode mode)
{
    if (!pu_process || !pu_process->isRunning())
        return false;

    QJsonObject req;
    req["method"_L1]  = "set_ch_mode"_L1;
    req["channel"_L1] = index;
    req["mode"_L1]    = static_cast<int>(mode);
    auto resp = pu_process->sendRequest(req);
    return !resp.contains("error"_L1) &&
           resp["result"_L1].toBool(false);
}

bool PythonPulseGenerator::setChDutyOn(const int index, const int pulses)
{
    if (!pu_process || !pu_process->isRunning())
        return false;

    QJsonObject req;
    req["method"_L1]  = "set_ch_duty_on"_L1;
    req["channel"_L1] = index;
    req["pulses"_L1]  = pulses;
    auto resp = pu_process->sendRequest(req);
    return !resp.contains("error"_L1) &&
           resp["result"_L1].toBool(false);
}

bool PythonPulseGenerator::setChDutyOff(const int index, const int pulses)
{
    if (!pu_process || !pu_process->isRunning())
        return false;

    QJsonObject req;
    req["method"_L1]  = "set_ch_duty_off"_L1;
    req["channel"_L1] = index;
    req["pulses"_L1]  = pulses;
    auto resp = pu_process->sendRequest(req);
    return !resp.contains("error"_L1) &&
           resp["result"_L1].toBool(false);
}

// ============================================================================
// Global set virtuals
// ============================================================================

bool PythonPulseGenerator::setHwPulseMode(PulseGenConfig::PGenMode mode)
{
    if (!pu_process || !pu_process->isRunning())
        return false;

    QJsonObject req;
    req["method"_L1] = "set_hw_pulse_mode"_L1;
    req["mode"_L1]   = static_cast<int>(mode);
    auto resp = pu_process->sendRequest(req);
    return !resp.contains("error"_L1) &&
           resp["result"_L1].toBool(false);
}

bool PythonPulseGenerator::setHwRepRate(double rr)
{
    if (!pu_process || !pu_process->isRunning())
        return false;

    QJsonObject req;
    req["method"_L1]   = "set_hw_rep_rate"_L1;
    req["rep_rate"_L1] = rr;
    auto resp = pu_process->sendRequest(req);
    return !resp.contains("error"_L1) &&
           resp["result"_L1].toBool(false);
}

bool PythonPulseGenerator::setHwPulseEnabled(bool en)
{
    if (!pu_process || !pu_process->isRunning())
        return false;

    QJsonObject req;
    req["method"_L1]  = "set_hw_pulse_enabled"_L1;
    req["enabled"_L1] = en;
    auto resp = pu_process->sendRequest(req);
    return !resp.contains("error"_L1) &&
           resp["result"_L1].toBool(false);
}

// ============================================================================
// Per-channel read virtuals
// ============================================================================

double PythonPulseGenerator::readChWidth(const int index)
{
    if (!pu_process || !pu_process->isRunning())
        return 0.0;

    QJsonObject req;
    req["method"_L1]  = "read_ch_width"_L1;
    req["channel"_L1] = index;
    auto resp = pu_process->sendRequest(req);

    if (resp.contains("error"_L1))
        return 0.0;

    return resp["result"_L1].toDouble(0.0);
}

double PythonPulseGenerator::readChDelay(const int index)
{
    if (!pu_process || !pu_process->isRunning())
        return 0.0;

    QJsonObject req;
    req["method"_L1]  = "read_ch_delay"_L1;
    req["channel"_L1] = index;
    auto resp = pu_process->sendRequest(req);

    if (resp.contains("error"_L1))
        return 0.0;

    return resp["result"_L1].toDouble(0.0);
}

PulseGenConfig::ActiveLevel PythonPulseGenerator::readChActiveLevel(const int index)
{
    if (!pu_process || !pu_process->isRunning())
        return PulseGenConfig::ActiveHigh;

    QJsonObject req;
    req["method"_L1]  = "read_ch_active_level"_L1;
    req["channel"_L1] = index;
    auto resp = pu_process->sendRequest(req);

    if (resp.contains("error"_L1))
        return PulseGenConfig::ActiveHigh;

    return static_cast<PulseGenConfig::ActiveLevel>(
        resp["result"_L1].toInt(
            static_cast<int>(PulseGenConfig::ActiveHigh)));
}

bool PythonPulseGenerator::readChEnabled(const int index)
{
    if (!pu_process || !pu_process->isRunning())
        return false;

    QJsonObject req;
    req["method"_L1]  = "read_ch_enabled"_L1;
    req["channel"_L1] = index;
    auto resp = pu_process->sendRequest(req);

    if (resp.contains("error"_L1))
        return false;

    return resp["result"_L1].toBool(false);
}

int PythonPulseGenerator::readChSynchCh(const int index)
{
    if (!pu_process || !pu_process->isRunning())
        return 0;

    QJsonObject req;
    req["method"_L1]  = "read_ch_sync_ch"_L1;
    req["channel"_L1] = index;
    auto resp = pu_process->sendRequest(req);

    if (resp.contains("error"_L1))
        return 0;

    return resp["result"_L1].toInt(0);
}

PulseGenConfig::ChannelMode PythonPulseGenerator::readChMode(const int index)
{
    if (!pu_process || !pu_process->isRunning())
        return PulseGenConfig::Normal;

    QJsonObject req;
    req["method"_L1]  = "read_ch_mode"_L1;
    req["channel"_L1] = index;
    auto resp = pu_process->sendRequest(req);

    if (resp.contains("error"_L1))
        return PulseGenConfig::Normal;

    return static_cast<PulseGenConfig::ChannelMode>(
        resp["result"_L1].toInt(
            static_cast<int>(PulseGenConfig::Normal)));
}

int PythonPulseGenerator::readChDutyOn(const int index)
{
    if (!pu_process || !pu_process->isRunning())
        return 1;

    QJsonObject req;
    req["method"_L1]  = "read_ch_duty_on"_L1;
    req["channel"_L1] = index;
    auto resp = pu_process->sendRequest(req);

    if (resp.contains("error"_L1))
        return 1;

    return resp["result"_L1].toInt(1);
}

int PythonPulseGenerator::readChDutyOff(const int index)
{
    if (!pu_process || !pu_process->isRunning())
        return 1;

    QJsonObject req;
    req["method"_L1]  = "read_ch_duty_off"_L1;
    req["channel"_L1] = index;
    auto resp = pu_process->sendRequest(req);

    if (resp.contains("error"_L1))
        return 1;

    return resp["result"_L1].toInt(1);
}

// ============================================================================
// Global read virtuals
// ============================================================================

PulseGenConfig::PGenMode PythonPulseGenerator::readHwPulseMode()
{
    if (!pu_process || !pu_process->isRunning())
        return PulseGenConfig::Continuous;

    QJsonObject req;
    req["method"_L1] = "read_hw_pulse_mode"_L1;
    auto resp = pu_process->sendRequest(req);

    if (resp.contains("error"_L1))
        return PulseGenConfig::Continuous;

    return static_cast<PulseGenConfig::PGenMode>(
        resp["result"_L1].toInt(
            static_cast<int>(PulseGenConfig::Continuous)));
}

double PythonPulseGenerator::readHwRepRate()
{
    if (!pu_process || !pu_process->isRunning())
        return 0.0;

    QJsonObject req;
    req["method"_L1] = "read_hw_rep_rate"_L1;
    auto resp = pu_process->sendRequest(req);

    if (resp.contains("error"_L1))
        return 0.0;

    return resp["result"_L1].toDouble(0.0);
}

bool PythonPulseGenerator::readHwPulseEnabled()
{
    if (!pu_process || !pu_process->isRunning())
        return false;

    QJsonObject req;
    req["method"_L1] = "read_hw_pulse_enabled"_L1;
    auto resp = pu_process->sendRequest(req);

    if (resp.contains("error"_L1))
        return false;

    return resp["result"_L1].toBool(false);
}
