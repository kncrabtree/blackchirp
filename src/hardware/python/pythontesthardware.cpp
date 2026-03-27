#include "pythontesthardware.h"
#include "pythonprocess.h"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QJsonObject>

#include <hardware/core/hardwareregistration.h>
#include <hardware/core/hardwareprofilemanager.h>

// ============================================================================
// Registration
// ============================================================================
REGISTER_HARDWARE_META(PythonTestHardware, "Python Test Hardware (QProcess proof-of-concept)")
REGISTER_HARDWARE_PROTOCOLS(PythonTestHardware, CommunicationProtocol::Tcp, CommunicationProtocol::Virtual)

// ============================================================================
// Constructor / Destructor
// ============================================================================
PythonTestHardware::PythonTestHardware(const QString &label, QObject *parent) :
    HardwareObject(QString(PythonTestHardware::staticMetaObject.className()),
                   QString(PythonTestHardware::staticMetaObject.className()),
                   label, parent)
{
    d_threaded = true;
    d_critical = false;

    setDefault(BC::Key::PythonTest::pythonScript, QString{});
    setDefault(BC::Key::PythonTest::pythonClass, QStringLiteral("TestHardware"));

    save();
}

PythonTestHardware::~PythonTestHardware()
{
    if (pu_process)
        pu_process->stop();
}

// ============================================================================
// initialize()
// ============================================================================
void PythonTestHardware::initialize()
{
    pu_process = std::make_unique<PythonProcess>(this);
    pu_process->setComm(p_comm);
    pu_process->setHardwareInfo(d_key, d_model);

    // Provide settings access via callbacks (SettingsStorage::set is protected,
    // but PythonTestHardware inherits SettingsStorage and can call set/get)
    pu_process->setSettingsCallbacks(
        [this](const QString &key, const QVariant &defaultVal) -> QVariant {
            return get(key, defaultVal);
        },
        [this](const QString &key, const QVariant &val) {
            set(key, val, true);
        }
    );

    connect(pu_process.get(), &PythonProcess::logMessage,
            this, &PythonTestHardware::logMessage);
}

// ============================================================================
// testConnection()
// ============================================================================
bool PythonTestHardware::testConnection()
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
bool PythonTestHardware::startPythonProcess()
{
    // Get script path from HardwareProfileManager (set in RuntimeHardwareConfigDialog)
    auto [hwType, label] = BC::Key::parseKey(d_key);
    QString scriptPath = HardwareProfileManager::instance().getPythonScriptPath(hwType, label);

    // Fall back to own SettingsStorage if not set in profile
    if (scriptPath.isEmpty())
        scriptPath = get<QString>(BC::Key::PythonTest::pythonScript);

    if (scriptPath.isEmpty()) {
        d_errorString = QStringLiteral("No Python script path configured");
        emit logMessage(QString("PythonTestHardware (%1): %2").arg(d_key, d_errorString),
                        LogHandler::Error);
        return false;
    }

    QString hostScript = findHostScript();
    if (hostScript.isEmpty()) {
        d_errorString = QStringLiteral("Cannot find python_hw_host.py");
        emit logMessage(QString("PythonTestHardware (%1): %2").arg(d_key, d_errorString),
                        LogHandler::Error);
        return false;
    }

    QString className = get<QString>(BC::Key::PythonTest::pythonClass);
    if (className.isEmpty())
        className = QStringLiteral("TestHardware");

    return pu_process->start(hostScript, scriptPath, className);
}

// ============================================================================
// findHostScript()
// ============================================================================
QString PythonTestHardware::findHostScript() const
{
    // Search in several locations for python_hw_host.py
    QStringList searchPaths = {
        QCoreApplication::applicationDirPath() + QStringLiteral("/python_hw_host.py"),
        QCoreApplication::applicationDirPath() + QStringLiteral("/../dev-docs/python_hw_host.py"),
        QCoreApplication::applicationDirPath() + QStringLiteral("/../../dev-docs/python_hw_host.py"),
        QCoreApplication::applicationDirPath() + QStringLiteral("/../share/blackchirp/python_hw_host.py"),
    };

    for (const auto &path : searchPaths) {
        if (QFile::exists(path))
            return path;
    }

    return {};
}

// ============================================================================
// readAuxData()
// ============================================================================
AuxDataStorage::AuxDataMap PythonTestHardware::readAuxData()
{
    emit logMessage(QString("readAuxData: pu_process=%1 isRunning=%2")
                    .arg(pu_process ? "valid" : "null")
                    .arg(pu_process && pu_process->isRunning() ? "true" : "false"),
                    LogHandler::Debug);

    if (!pu_process || !pu_process->isRunning())
        return {};

    QJsonObject req;
    req[QStringLiteral("method")] = QStringLiteral("read_aux_data");
    auto resp = pu_process->sendRequest(req);

    emit logMessage(QString("readAuxData: response keys=%1")
                    .arg(resp.keys().join(",")),
                    LogHandler::Debug);

    if (resp.contains(QStringLiteral("error"))) {
        emit logMessage(QString("readAuxData: error=%1")
                        .arg(resp[QStringLiteral("error")].toString()),
                        LogHandler::Debug);
        return {};
    }

    return parseAuxDataResult(resp);
}

// ============================================================================
// readValidationData()
// ============================================================================
AuxDataStorage::AuxDataMap PythonTestHardware::readValidationData()
{
    if (!pu_process || !pu_process->isRunning())
        return {};

    QJsonObject req;
    req[QStringLiteral("method")] = QStringLiteral("read_validation_data");
    auto resp = pu_process->sendRequest(req);

    if (resp.contains(QStringLiteral("error")))
        return {};

    return parseAuxDataResult(resp);
}

// ============================================================================
// prepareForExperiment()
// ============================================================================
bool PythonTestHardware::prepareForExperiment(Experiment &exp)
{
    emit logMessage(QString("prepareForExperiment: exp.d_number=%1 process=%2")
                    .arg(exp.d_number)
                    .arg(pu_process && pu_process->isRunning() ? "running" : "not running"),
                    LogHandler::Debug);

    if (!pu_process || !pu_process->isRunning())
        return true;

    QJsonObject req;
    req[QStringLiteral("method")] = QStringLiteral("prepare_for_experiment");

    QJsonObject config;
    config[QStringLiteral("number")] = exp.d_number;
    req[QStringLiteral("config")] = config;

    auto resp = pu_process->sendRequest(req);
    if (resp.contains(QStringLiteral("error"))) {
        d_errorString = resp[QStringLiteral("error")].toString();
        emit logMessage(QString("PythonTestHardware (%1): prepareForExperiment error: %2")
                            .arg(d_key, d_errorString),
                        LogHandler::Error);
        return false;
    }
    return resp[QStringLiteral("result")].toBool(true);
}

// ============================================================================
// beginAcquisition()
// ============================================================================
void PythonTestHardware::beginAcquisition()
{
    if (!pu_process || !pu_process->isRunning())
        return;

    QJsonObject req;
    req[QStringLiteral("method")] = QStringLiteral("begin_acquisition");
    pu_process->sendRequest(req);
}

// ============================================================================
// endAcquisition()
// ============================================================================
void PythonTestHardware::endAcquisition()
{
    if (!pu_process || !pu_process->isRunning())
        return;

    QJsonObject req;
    req[QStringLiteral("method")] = QStringLiteral("end_acquisition");
    pu_process->sendRequest(req);
}

// ============================================================================
// sleep()
// ============================================================================
void PythonTestHardware::sleep(bool b)
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
void PythonTestHardware::readSettings()
{
    emit logMessage(QString("readSettings: hot-reload triggered, process=%1")
                    .arg(pu_process ? "valid" : "null"),
                    LogHandler::Debug);

    if (pu_process) {
        pu_process->stop();
        startPythonProcess();
    }
}

QStringList PythonTestHardware::forbiddenKeys() const
{
    return {BC::Key::HW::commType, BC::Key::HW::model,
            BC::Key::PythonTest::pythonScript, BC::Key::PythonTest::pythonClass};
}

// ============================================================================
// parseAuxDataResult()
// ============================================================================
AuxDataStorage::AuxDataMap PythonTestHardware::parseAuxDataResult(const QJsonObject &response)
{
    AuxDataStorage::AuxDataMap out;
    QJsonObject result = response[QStringLiteral("result")].toObject();
    for (auto it = result.begin(); it != result.end(); ++it)
        out[it.key()] = QVariant(it.value().toDouble());
    return out;
}
