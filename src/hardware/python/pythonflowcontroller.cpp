#include "pythonflowcontroller.h"

#include <QJsonObject>

#include <hardware/core/hardwareregistration.h>

// ============================================================================
// Registration
// ============================================================================
REGISTER_HARDWARE_META(PythonFlowController, "Python Flow Controller (user-defined Python script)")
REGISTER_HARDWARE_PROTOCOLS(PythonFlowController, CommunicationProtocol::Rs232, CommunicationProtocol::Tcp, CommunicationProtocol::Gpib, CommunicationProtocol::Custom, CommunicationProtocol::Virtual)
REGISTER_HARDWARE_SETTINGS(PythonFlowController,
    {BC::Key::Flow::pUnits, "Pressure Units",
     "Units for pressure reading display.",
     QString("Torr"), QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Flow::pMax, "Max Pressure",
     "Full-scale pressure for display scaling.",
     1000.0, 0.0, QVariant{}, HwSettingPriority::Optional}
)
REGISTER_HARDWARE_ARRAY(PythonFlowController, BC::Key::Flow::channels,
    "Flow Channels", "Per-channel mass flow controller configuration.", HwSettingPriority::Important)
REGISTER_HARDWARE_ARRAY_ENTRY(PythonFlowController, BC::Key::Flow::channels,
    {{BC::Key::Flow::chUnits, QString("sccm")}, {BC::Key::Flow::chMax, 10000.0}, {BC::Key::Flow::chDecimals, 3}})
REGISTER_HARDWARE_ARRAY_ENTRY(PythonFlowController, BC::Key::Flow::channels,
    {{BC::Key::Flow::chUnits, QString("sccm")}, {BC::Key::Flow::chMax, 10000.0}, {BC::Key::Flow::chDecimals, 3}})
REGISTER_HARDWARE_ARRAY_ENTRY(PythonFlowController, BC::Key::Flow::channels,
    {{BC::Key::Flow::chUnits, QString("sccm")}, {BC::Key::Flow::chMax, 10000.0}, {BC::Key::Flow::chDecimals, 3}})
REGISTER_HARDWARE_ARRAY_ENTRY(PythonFlowController, BC::Key::Flow::channels,
    {{BC::Key::Flow::chUnits, QString("sccm")}, {BC::Key::Flow::chMax, 10000.0}, {BC::Key::Flow::chDecimals, 3}})

// ============================================================================
// Constructor / Destructor
// ============================================================================
PythonFlowController::PythonFlowController(const QString &label, QObject *parent) :
    FlowController(QString(PythonFlowController::staticMetaObject.className()), label, parent),
    PythonHardwareBase(d_key, d_model)
{
    d_threaded = true;

    save();
}

// ============================================================================
// fcInitialize()
// ============================================================================
void PythonFlowController::fcInitialize()
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
// fcTestConnection()
// ============================================================================
bool PythonFlowController::fcTestConnection()
{
    if (!testPythonConnection(p_comm)) {
        d_errorString = pythonErrorString();
        return false;
    }
    return true;
}

// ============================================================================
// fcReadSettings()
// ============================================================================
void PythonFlowController::fcReadSettings()
{
    pythonReadSettings();
}

// ============================================================================
// hw* pure virtual implementations
// ============================================================================

void PythonFlowController::hwSetPressureControlMode(bool enabled)
{
    if (!pu_process || !pu_process->isRunning())
        return;

    QJsonObject req;
    req["method"_L1]  = "hw_set_pressure_control_mode"_L1;
    req["enabled"_L1] = enabled;
    pu_process->sendRequest(req);
}

void PythonFlowController::hwSetFlowSetpoint(const int ch, const double val)
{
    if (!pu_process || !pu_process->isRunning())
        return;

    QJsonObject req;
    req["method"_L1]  = "hw_set_flow_setpoint"_L1;
    req["channel"_L1] = ch;
    req["value"_L1]   = val;
    pu_process->sendRequest(req);
}

void PythonFlowController::hwSetChannelEnabled(const int ch, const bool en)
{
    if (!pu_process || !pu_process->isRunning())
        return;

    QJsonObject req;
    req["method"_L1]  = "hw_set_channel_enabled"_L1;
    req["channel"_L1] = ch;
    req["enabled"_L1] = en;
    pu_process->sendRequest(req);
}

void PythonFlowController::hwSetPressureSetpoint(const double val)
{
    if (!pu_process || !pu_process->isRunning())
        return;

    QJsonObject req;
    req["method"_L1] = "hw_set_pressure_setpoint"_L1;
    req["value"_L1]  = val;
    pu_process->sendRequest(req);
}

double PythonFlowController::hwReadFlowSetpoint(const int ch)
{
    if (!pu_process || !pu_process->isRunning())
        return -1.0;

    QJsonObject req;
    req["method"_L1]  = "hw_read_flow_setpoint"_L1;
    req["channel"_L1] = ch;
    auto resp = pu_process->sendRequest(req);

    if (resp.contains("error"_L1))
        return -1.0;

    return resp["result"_L1].toDouble(-1.0);
}

double PythonFlowController::hwReadPressureSetpoint()
{
    if (!pu_process || !pu_process->isRunning())
        return -1.0;

    QJsonObject req;
    req["method"_L1] = "hw_read_pressure_setpoint"_L1;
    auto resp = pu_process->sendRequest(req);

    if (resp.contains("error"_L1))
        return -1.0;

    return resp["result"_L1].toDouble(-1.0);
}

double PythonFlowController::hwReadFlow(const int ch)
{
    if (!pu_process || !pu_process->isRunning())
        return -1.0;

    QJsonObject req;
    req["method"_L1]  = "hw_read_flow"_L1;
    req["channel"_L1] = ch;
    auto resp = pu_process->sendRequest(req);

    if (resp.contains("error"_L1))
        return -1.0;

    return resp["result"_L1].toDouble(-1.0);
}

double PythonFlowController::hwReadPressure()
{
    if (!pu_process || !pu_process->isRunning())
        return -1.0;

    QJsonObject req;
    req["method"_L1] = "hw_read_pressure"_L1;
    auto resp = pu_process->sendRequest(req);

    if (resp.contains("error"_L1))
        return -1.0;

    return resp["result"_L1].toDouble(-1.0);
}

int PythonFlowController::hwReadPressureControlMode()
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
