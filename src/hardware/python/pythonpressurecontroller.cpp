#include "pythonpressurecontroller.h"

#include <QJsonObject>
#include <cmath>

#include <hardware/core/hardwareregistration.h>
#include <data/settings/hardwarekeys.h>
#include <data/bcglobals.h>

// ============================================================================
// Registration
// ============================================================================
REGISTER_HARDWARE_META(PythonPressureController, "Python Pressure Controller (user-defined Python script)")
REGISTER_HARDWARE_PROTOCOLS(PythonPressureController, CommunicationProtocol::Rs232, CommunicationProtocol::Tcp, CommunicationProtocol::Virtual)
REGISTER_HARDWARE_PARAMS(PythonPressureController)

// ============================================================================
// Constructor / Destructor
// ============================================================================
PythonPressureController::PythonPressureController(const QString &label, QObject *parent) :
    PressureController(
        QString(PythonPressureController::staticMetaObject.className()),
        label,
        [&label]() -> bool {
            SettingsStorage s(BC::Key::hwKey(
                                  QString(PressureController::staticMetaObject.className()),
                                  label)
                              );
            auto ro = s.get(BC::Key::PController::readOnly, false);
            return ro;
        }(),
        parent),
    PythonHardwareBase(d_key, d_model)
{
    d_threaded = true;

    setDefault(BC::Key::PController::min, -1.0);
    setDefault(BC::Key::PController::max, 20.0);
    setDefault(BC::Key::PController::decimals, 4);
    setDefault(BC::Key::PController::units, QString("Torr"));
    setDefault(BC::Key::PController::readInterval, 200);
    setDefault(BC::Key::PController::hasValve, true);

    save();
}

// ============================================================================
// configParams()
// ============================================================================
QVector<HwConfigParam> PythonPressureController::configParams()
{
    using namespace BC::Key::PController;
    return {
        { readOnly, QStringLiteral("Read Only"), QVariant(false), 0, 0 },
    };
}

// ============================================================================
// pcInitialize()
// ============================================================================
void PythonPressureController::pcInitialize()
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
            this, &PythonPressureController::logMessage);
}

// ============================================================================
// pcTestConnection()
// ============================================================================
bool PythonPressureController::pcTestConnection()
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
void PythonPressureController::readSettings()
{
    pythonReadSettings();
}

// ============================================================================
// sleep()
// ============================================================================
void PythonPressureController::sleep(bool b)
{
    pythonSleep(b);
}

// ============================================================================
// forbiddenKeys()
// ============================================================================
QStringList PythonPressureController::forbiddenKeys() const
{
    auto keys = pythonForbiddenKeys();
    keys << BC::Key::PController::readOnly;
    return keys;
}

// ============================================================================
// hw* pure virtual implementations
// ============================================================================

double PythonPressureController::hwReadPressure()
{
    if (!pu_process || !pu_process->isRunning())
        return std::nan("");

    QJsonObject req;
    req[QStringLiteral("method")] = QStringLiteral("hw_read_pressure");
    auto resp = pu_process->sendRequest(req);

    if (resp.contains(QStringLiteral("error")))
        return std::nan("");

    return resp[QStringLiteral("result")].toDouble(std::nan(""));
}

double PythonPressureController::hwSetPressureSetpoint(const double val)
{
    if (!pu_process || !pu_process->isRunning())
        return std::nan("");

    QJsonObject req;
    req[QStringLiteral("method")] = QStringLiteral("hw_set_pressure_setpoint");
    req[QStringLiteral("value")]  = val;
    auto resp = pu_process->sendRequest(req);

    if (resp.contains(QStringLiteral("error")))
        return std::nan("");

    return resp[QStringLiteral("result")].toDouble(val);
}

double PythonPressureController::hwReadPressureSetpoint()
{
    if (!pu_process || !pu_process->isRunning())
        return std::nan("");

    QJsonObject req;
    req[QStringLiteral("method")] = QStringLiteral("hw_read_pressure_setpoint");
    auto resp = pu_process->sendRequest(req);

    if (resp.contains(QStringLiteral("error")))
        return std::nan("");

    return resp[QStringLiteral("result")].toDouble(std::nan(""));
}

void PythonPressureController::hwSetPressureControlMode(bool enabled)
{
    if (!pu_process || !pu_process->isRunning())
        return;

    QJsonObject req;
    req[QStringLiteral("method")]  = QStringLiteral("hw_set_pressure_control_mode");
    req[QStringLiteral("enabled")] = enabled;
    pu_process->sendRequest(req);
}

int PythonPressureController::hwReadPressureControlMode()
{
    if (!pu_process || !pu_process->isRunning())
        return -1;

    QJsonObject req;
    req[QStringLiteral("method")] = QStringLiteral("hw_read_pressure_control_mode");
    auto resp = pu_process->sendRequest(req);

    if (resp.contains(QStringLiteral("error")))
        return -1;

    return resp[QStringLiteral("result")].toInt(-1);
}

void PythonPressureController::hwOpenGateValve()
{
    if (!pu_process || !pu_process->isRunning())
        return;

    QJsonObject req;
    req[QStringLiteral("method")] = QStringLiteral("hw_open_gate_valve");
    pu_process->sendRequest(req);
}

void PythonPressureController::hwCloseGateValve()
{
    if (!pu_process || !pu_process->isRunning())
        return;

    QJsonObject req;
    req[QStringLiteral("method")] = QStringLiteral("hw_close_gate_valve");
    pu_process->sendRequest(req);
}
