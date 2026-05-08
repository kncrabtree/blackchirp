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
REGISTER_HARDWARE_PROTOCOLS(PythonPressureController, CommunicationProtocol::Rs232, CommunicationProtocol::Tcp, CommunicationProtocol::Gpib, CommunicationProtocol::Custom, CommunicationProtocol::Virtual)
REGISTER_HARDWARE_SETTINGS(PythonPressureController,
    {BC::Key::PController::readOnly, "Read Only",
     "Device is a read-only pressure gauge (no valve control).",
     false, QVariant{}, QVariant{}, HwSettingPriority::Required}
)

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

    save();
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
// pcReadSettings()
// ============================================================================
void PythonPressureController::pcReadSettings()
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
// hw* pure virtual implementations
// ============================================================================

double PythonPressureController::hwReadPressure()
{
    if (!pu_process || !pu_process->isRunning())
        return std::nan("");

    QJsonObject req;
    req["method"_L1] = "hw_read_pressure"_L1;
    auto resp = pu_process->sendRequest(req);

    if (resp.contains("error"_L1))
        return std::nan("");

    return resp["result"_L1].toDouble(std::nan(""));
}

double PythonPressureController::hwSetPressureSetpoint(const double val)
{
    if (!pu_process || !pu_process->isRunning())
        return std::nan("");

    QJsonObject req;
    req["method"_L1] = "hw_set_pressure_setpoint"_L1;
    req["value"_L1]  = val;
    auto resp = pu_process->sendRequest(req);

    if (resp.contains("error"_L1))
        return std::nan("");

    return resp["result"_L1].toDouble(val);
}

double PythonPressureController::hwReadPressureSetpoint()
{
    if (!pu_process || !pu_process->isRunning())
        return std::nan("");

    QJsonObject req;
    req["method"_L1] = "hw_read_pressure_setpoint"_L1;
    auto resp = pu_process->sendRequest(req);

    if (resp.contains("error"_L1))
        return std::nan("");

    return resp["result"_L1].toDouble(std::nan(""));
}

void PythonPressureController::hwSetPressureControlMode(bool enabled)
{
    if (!pu_process || !pu_process->isRunning())
        return;

    QJsonObject req;
    req["method"_L1]  = "hw_set_pressure_control_mode"_L1;
    req["enabled"_L1] = enabled;
    pu_process->sendRequest(req);
}

int PythonPressureController::hwReadPressureControlMode()
{
    if (!pu_process || !pu_process->isRunning())
        return -1;

    QJsonObject req;
    req["method"_L1] = "hw_read_pressure_control_mode"_L1;
    auto resp = pu_process->sendRequest(req);

    if (resp.contains("error"_L1))
        return -1;

    return resp["result"_L1].toInt(-1);
}

void PythonPressureController::hwOpenGateValve()
{
    if (!pu_process || !pu_process->isRunning())
        return;

    QJsonObject req;
    req["method"_L1] = "hw_open_gate_valve"_L1;
    pu_process->sendRequest(req);
}

void PythonPressureController::hwCloseGateValve()
{
    if (!pu_process || !pu_process->isRunning())
        return;

    QJsonObject req;
    req["method"_L1] = "hw_close_gate_valve"_L1;
    pu_process->sendRequest(req);
}
