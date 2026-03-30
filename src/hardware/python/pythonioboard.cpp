#include "pythonioboard.h"

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
REGISTER_HARDWARE_META(PythonIOBoard, "Python IO Board (user-defined Python script)")
REGISTER_HARDWARE_PROTOCOLS(PythonIOBoard, CommunicationProtocol::Rs232, CommunicationProtocol::Tcp, CommunicationProtocol::Virtual)

// ============================================================================
// Constructor / Destructor
// ============================================================================
PythonIOBoard::PythonIOBoard(const QString &label, QObject *parent) :
    IOBoard(QString(PythonIOBoard::staticMetaObject.className()), label, parent)
{
    d_threaded = true;

    setDefault(BC::Key::PythonIOBoard::pythonScript, QString{});
    setDefault(BC::Key::PythonIOBoard::pythonClass, QStringLiteral("IOBoardDriver"));

    using namespace BC::Key::Digi;
    setDefault(numAnalogChannels, 0);
    setDefault(numDigitalChannels, 0);

    save();
}

PythonIOBoard::~PythonIOBoard()
{
    if (pu_process)
        pu_process->stop();
}

// ============================================================================
// initialize()
// ============================================================================
void PythonIOBoard::initialize()
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
            this, &PythonIOBoard::logMessage);
}

// ============================================================================
// testConnection()
// ============================================================================
bool PythonIOBoard::testConnection()
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
bool PythonIOBoard::startPythonProcess()
{
    auto [hwType, label] = BC::Key::parseKey(d_key);
    QString scriptPath = HardwareProfileManager::instance().getPythonScriptPath(hwType, label);

    if (scriptPath.isEmpty())
        scriptPath = get<QString>(BC::Key::PythonIOBoard::pythonScript);

    if (scriptPath.isEmpty()) {
        d_errorString = QStringLiteral("No Python script path configured");
        emit logMessage(QString("PythonIOBoard (%1): %2").arg(d_key, d_errorString),
                        LogHandler::Error);
        return false;
    }

    QString hostScript = findHostScript();
    if (hostScript.isEmpty()) {
        d_errorString = QStringLiteral("Cannot find python_hw_host.py");
        emit logMessage(QString("PythonIOBoard (%1): %2").arg(d_key, d_errorString),
                        LogHandler::Error);
        return false;
    }

    QString className = get<QString>(BC::Key::PythonIOBoard::pythonClass);
    if (className.isEmpty())
        className = QStringLiteral("IOBoardDriver");

    return pu_process->start(hostScript, scriptPath, className);
}

// ============================================================================
// findHostScript()
// ============================================================================
QString PythonIOBoard::findHostScript() const
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
// readAnalogChannels()
// ============================================================================
std::map<int, double> PythonIOBoard::readAnalogChannels()
{
    if (!pu_process || !pu_process->isRunning())
        return {};

    QJsonObject req;
    req[QStringLiteral("method")] = QStringLiteral("read_analog_channels");
    auto resp = pu_process->sendRequest(req);

    if (resp.contains(QStringLiteral("error")))
        return {};

    std::map<int, double> out;
    QJsonObject result = resp[QStringLiteral("result")].toObject();
    for (auto it = result.begin(); it != result.end(); ++it)
        out[it.key().toInt()] = it.value().toDouble();

    return out;
}

// ============================================================================
// readDigitalChannels()
// ============================================================================
std::map<int, bool> PythonIOBoard::readDigitalChannels()
{
    if (!pu_process || !pu_process->isRunning())
        return {};

    QJsonObject req;
    req[QStringLiteral("method")] = QStringLiteral("read_digital_channels");
    auto resp = pu_process->sendRequest(req);

    if (resp.contains(QStringLiteral("error")))
        return {};

    std::map<int, bool> out;
    QJsonObject result = resp[QStringLiteral("result")].toObject();
    for (auto it = result.begin(); it != result.end(); ++it)
        out[it.key().toInt()] = it.value().toBool();

    return out;
}

// ============================================================================
// sleep()
// ============================================================================
void PythonIOBoard::sleep(bool b)
{
    if (!pu_process || !pu_process->isRunning())
        return;

    QJsonObject req;
    req[QStringLiteral("method")] = QStringLiteral("sleep");
    req[QStringLiteral("sleeping")] = b;
    pu_process->sendRequest(req);
}

// ============================================================================
// readSettings() -- hot-reload
// ============================================================================
void PythonIOBoard::readSettings()
{
    if (pu_process) {
        pu_process->stop();
        startPythonProcess();
    }
}

// ============================================================================
// forbiddenKeys()
// ============================================================================
QStringList PythonIOBoard::forbiddenKeys() const
{
    return {BC::Key::HW::commType, BC::Key::HW::model,
            BC::Key::PythonIOBoard::pythonScript, BC::Key::PythonIOBoard::pythonClass,
            BC::Key::Digi::numAnalogChannels, BC::Key::Digi::numDigitalChannels};
}

#endif // BC_PYTHON_HARDWARE
