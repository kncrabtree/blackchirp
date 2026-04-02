#include "pythonflowcontroller.h"

#ifdef BC_PYTHON_HARDWARE

#include <QJsonObject>

#include <hardware/core/hardwareregistration.h>

// ============================================================================
// Registration
// ============================================================================
REGISTER_HARDWARE_META(PythonFlowController, "Python Flow Controller (user-defined Python script)")
REGISTER_HARDWARE_PROTOCOLS(PythonFlowController, CommunicationProtocol::Rs232, CommunicationProtocol::Tcp, CommunicationProtocol::Virtual)

// ============================================================================
// Constructor / Destructor
// ============================================================================
PythonFlowController::PythonFlowController(const QString &label, QObject *parent) :
    FlowController(QString(PythonFlowController::staticMetaObject.className()), label, parent),
    PythonHardwareBase(d_key, d_model)
{
    d_threaded = true;

    if (!containsArray(BC::Key::Flow::channels))
    {
        std::vector<SettingsMap> l;
        int ch = get(BC::Key::Flow::flowChannels, 4);
        l.reserve(ch);
        for (int i = 0; i < ch; ++i)
            l.push_back({{BC::Key::Flow::chUnits, QString("sccm")},
                         {BC::Key::Flow::chMax,   10000.0},
                         {BC::Key::Flow::chDecimals, 3}});

        setArray(BC::Key::Flow::channels, l, true);
    }

    setDefault(BC::Key::Flow::pUnits,   QString("Torr"));
    setDefault(BC::Key::Flow::pMax,     1000.0);
    setDefault(BC::Key::Flow::pDec,     3);

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

    connect(pu_process.get(), &PythonProcess::logMessage,
            this, &PythonFlowController::logMessage);
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
// readSettings()
// ============================================================================
void PythonFlowController::readSettings()
{
    pythonReadSettings();
}

// ============================================================================
// forbiddenKeys()
// ============================================================================
QStringList PythonFlowController::forbiddenKeys() const
{
    return FlowController::forbiddenKeys() + pythonForbiddenKeys();
}

// ============================================================================
// hw* pure virtual implementations
// ============================================================================

void PythonFlowController::hwSetPressureControlMode(bool enabled)
{
    if (!pu_process || !pu_process->isRunning())
        return;

    QJsonObject req;
    req[QStringLiteral("method")]  = QStringLiteral("hw_set_pressure_control_mode");
    req[QStringLiteral("enabled")] = enabled;
    pu_process->sendRequest(req);
}

void PythonFlowController::hwSetFlowSetpoint(const int ch, const double val)
{
    if (!pu_process || !pu_process->isRunning())
        return;

    QJsonObject req;
    req[QStringLiteral("method")]  = QStringLiteral("hw_set_flow_setpoint");
    req[QStringLiteral("channel")] = ch;
    req[QStringLiteral("value")]   = val;
    pu_process->sendRequest(req);
}

void PythonFlowController::hwSetPressureSetpoint(const double val)
{
    if (!pu_process || !pu_process->isRunning())
        return;

    QJsonObject req;
    req[QStringLiteral("method")] = QStringLiteral("hw_set_pressure_setpoint");
    req[QStringLiteral("value")]  = val;
    pu_process->sendRequest(req);
}

double PythonFlowController::hwReadFlowSetpoint(const int ch)
{
    if (!pu_process || !pu_process->isRunning())
        return -1.0;

    QJsonObject req;
    req[QStringLiteral("method")]  = QStringLiteral("hw_read_flow_setpoint");
    req[QStringLiteral("channel")] = ch;
    auto resp = pu_process->sendRequest(req);

    if (resp.contains(QStringLiteral("error")))
        return -1.0;

    return resp[QStringLiteral("result")].toDouble(-1.0);
}

double PythonFlowController::hwReadPressureSetpoint()
{
    if (!pu_process || !pu_process->isRunning())
        return -1.0;

    QJsonObject req;
    req[QStringLiteral("method")] = QStringLiteral("hw_read_pressure_setpoint");
    auto resp = pu_process->sendRequest(req);

    if (resp.contains(QStringLiteral("error")))
        return -1.0;

    return resp[QStringLiteral("result")].toDouble(-1.0);
}

double PythonFlowController::hwReadFlow(const int ch)
{
    if (!pu_process || !pu_process->isRunning())
        return -1.0;

    QJsonObject req;
    req[QStringLiteral("method")]  = QStringLiteral("hw_read_flow");
    req[QStringLiteral("channel")] = ch;
    auto resp = pu_process->sendRequest(req);

    if (resp.contains(QStringLiteral("error")))
        return -1.0;

    return resp[QStringLiteral("result")].toDouble(-1.0);
}

double PythonFlowController::hwReadPressure()
{
    if (!pu_process || !pu_process->isRunning())
        return -1.0;

    QJsonObject req;
    req[QStringLiteral("method")] = QStringLiteral("hw_read_pressure");
    auto resp = pu_process->sendRequest(req);

    if (resp.contains(QStringLiteral("error")))
        return -1.0;

    return resp[QStringLiteral("result")].toDouble(-1.0);
}

int PythonFlowController::hwReadPressureControlMode()
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

#endif // BC_PYTHON_HARDWARE
