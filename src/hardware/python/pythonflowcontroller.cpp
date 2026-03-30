#include "pythonflowcontroller.h"

#ifdef BC_PYTHON_HARDWARE

#include "pythonprocess.h"

#include <QCoreApplication>
#include <QFile>
#include <QJsonObject>

#include <hardware/core/hardwareregistration.h>
#include <hardware/core/hardwareprofilemanager.h>

// ============================================================================
// Registration
// ============================================================================
REGISTER_HARDWARE_META(PythonFlowController, "Python Flow Controller (user-defined Python script)")
REGISTER_HARDWARE_PROTOCOLS(PythonFlowController, CommunicationProtocol::Rs232, CommunicationProtocol::Tcp, CommunicationProtocol::Virtual)

// ============================================================================
// Constructor / Destructor
// ============================================================================
PythonFlowController::PythonFlowController(const QString &label, QObject *parent) :
    FlowController(QString(PythonFlowController::staticMetaObject.className()), label, parent)
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

    setDefault(BC::Key::PythonFlowController::pythonScript, QString{});
    setDefault(BC::Key::PythonFlowController::pythonClass,  QStringLiteral("FlowControllerDriver"));

    save();
}

PythonFlowController::~PythonFlowController()
{
    if (pu_process)
        pu_process->stop();
}

// ============================================================================
// fcInitialize()
// ============================================================================
void PythonFlowController::fcInitialize()
{
    pu_process = std::make_unique<PythonProcess>(this);
    pu_process->setComm(p_comm);
    pu_process->setHardwareInfo(d_key, d_model);

    pu_process->setSettingsCallbacks(
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
    if (!pu_process->isRunning()) {
        if (!startPythonProcess())
            return false;
    }

    // Update comm in case protocol was reconfigured
    pu_process->setComm(p_comm);

    QJsonObject req;
    req[QStringLiteral("method")] = QStringLiteral("test_connection");
    auto resp = pu_process->sendRequest(req);

    if (resp.contains(QStringLiteral("error"))) {
        d_errorString = resp[QStringLiteral("error")].toString();
        return false;
    }
    return resp[QStringLiteral("result")].toBool(false);
}

// ============================================================================
// startPythonProcess()
// ============================================================================
bool PythonFlowController::startPythonProcess()
{
    auto [hwType, label] = BC::Key::parseKey(d_key);
    QString scriptPath = HardwareProfileManager::instance().getPythonScriptPath(hwType, label);

    if (scriptPath.isEmpty())
        scriptPath = get<QString>(BC::Key::PythonFlowController::pythonScript);

    if (scriptPath.isEmpty()) {
        d_errorString = QStringLiteral("No Python script path configured");
        emit logMessage(QString("PythonFlowController (%1): %2").arg(d_key, d_errorString),
                        LogHandler::Error);
        return false;
    }

    QString hostScript = findHostScript();
    if (hostScript.isEmpty()) {
        d_errorString = QStringLiteral("Cannot find python_hw_host.py");
        emit logMessage(QString("PythonFlowController (%1): %2").arg(d_key, d_errorString),
                        LogHandler::Error);
        return false;
    }

    QString className = get<QString>(BC::Key::PythonFlowController::pythonClass);
    if (className.isEmpty())
        className = QStringLiteral("FlowControllerDriver");

    return pu_process->start(hostScript, scriptPath, className);
}

// ============================================================================
// findHostScript()
// ============================================================================
QString PythonFlowController::findHostScript() const
{
    QStringList searchPaths = {
        QCoreApplication::applicationDirPath() + QStringLiteral("/python_hw_host.py"),
        QCoreApplication::applicationDirPath() + QStringLiteral("/../share/blackchirp/python_hw_host.py"),
    };

    for (const auto &path : searchPaths) {
        if (QFile::exists(path))
            return path;
    }

    return {};
}

// ============================================================================
// readSettings()
// ============================================================================
void PythonFlowController::readSettings()
{
    if (pu_process && pu_process->isRunning()) {
        QJsonObject req;
        req[QStringLiteral("method")] = QStringLiteral("read_settings");
        pu_process->sendRequest(req);
    }
}

// ============================================================================
// forbiddenKeys()
// ============================================================================
QStringList PythonFlowController::forbiddenKeys() const
{
    QStringList keys = FlowController::forbiddenKeys();
    keys << BC::Key::PythonFlowController::pythonScript
         << BC::Key::PythonFlowController::pythonClass;
    return keys;
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
