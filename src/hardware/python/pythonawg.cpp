#include "pythonawg.h"

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
REGISTER_HARDWARE_META(PythonAwg, "Python AWG (user-defined Python script)")
REGISTER_HARDWARE_PROTOCOLS(PythonAwg, CommunicationProtocol::Rs232, CommunicationProtocol::Tcp, CommunicationProtocol::Virtual)

// ============================================================================
// Constructor / Destructor
// ============================================================================
PythonAwg::PythonAwg(const QString &label, QObject *parent) :
    AWG(QString(PythonAwg::staticMetaObject.className()), label, parent)
{
    d_threaded = true;
    d_critical = false;

    setDefault(BC::Key::PythonAwg::pythonScript, QString{});
    setDefault(BC::Key::PythonAwg::pythonClass, QStringLiteral("AwgDriver"));

    save();
}

PythonAwg::~PythonAwg()
{
    if (pu_process)
        pu_process->stop();
}

// ============================================================================
// initialize()
// ============================================================================
void PythonAwg::initialize()
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
            this, &PythonAwg::logMessage);
}

// ============================================================================
// testConnection()
// ============================================================================
bool PythonAwg::testConnection()
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
bool PythonAwg::startPythonProcess()
{
    auto [hwType, label] = BC::Key::parseKey(d_key);
    QString scriptPath = HardwareProfileManager::instance().getPythonScriptPath(hwType, label);

    if (scriptPath.isEmpty())
        scriptPath = get<QString>(BC::Key::PythonAwg::pythonScript);

    if (scriptPath.isEmpty()) {
        d_errorString = QStringLiteral("No Python script path configured");
        emit logMessage(QString("PythonAwg (%1): %2").arg(d_key, d_errorString),
                        LogHandler::Error);
        return false;
    }

    QString hostScript = findHostScript();
    if (hostScript.isEmpty()) {
        d_errorString = QStringLiteral("Cannot find python_hw_host.py");
        emit logMessage(QString("PythonAwg (%1): %2").arg(d_key, d_errorString),
                        LogHandler::Error);
        return false;
    }

    QString className = get<QString>(BC::Key::PythonAwg::pythonClass);
    if (className.isEmpty())
        className = QStringLiteral("AwgDriver");

    return pu_process->start(hostScript, scriptPath, className);
}

// ============================================================================
// findHostScript()
// ============================================================================
QString PythonAwg::findHostScript() const
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
// readAuxData()
// ============================================================================
AuxDataStorage::AuxDataMap PythonAwg::readAuxData()
{
    if (!pu_process || !pu_process->isRunning())
        return {};

    QJsonObject req;
    req[QStringLiteral("method")] = QStringLiteral("read_aux_data");
    auto resp = pu_process->sendRequest(req);

    if (resp.contains(QStringLiteral("error")))
        return {};

    return parseAuxDataResult(resp);
}

// ============================================================================
// readValidationData()
// ============================================================================
AuxDataStorage::AuxDataMap PythonAwg::readValidationData()
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
bool PythonAwg::prepareForExperiment(Experiment &exp)
{
    if (!pu_process || !pu_process->isRunning())
        return true;

    auto auxData = readAuxData();
    for (auto it = auxData.cbegin(); it != auxData.cend(); ++it)
        exp.auxData()->registerKey(d_key, it->first);

    auto valData = readValidationData();
    for (auto it = valData.cbegin(); it != valData.cend(); ++it)
        exp.auxData()->registerKey(d_key, it->first);

    QJsonObject req;
    req[QStringLiteral("method")] = QStringLiteral("prepare_for_experiment");

    QJsonObject config;
    config[QStringLiteral("number")] = exp.d_number;
    req[QStringLiteral("config")] = config;

    auto resp = pu_process->sendRequest(req);
    if (resp.contains(QStringLiteral("error"))) {
        d_errorString = resp[QStringLiteral("error")].toString();
        emit logMessage(QString("PythonAwg (%1): prepareForExperiment error: %2")
                            .arg(d_key, d_errorString),
                        LogHandler::Error);
        return false;
    }
    return resp[QStringLiteral("result")].toBool(true);
}

// ============================================================================
// beginAcquisition()
// ============================================================================
void PythonAwg::beginAcquisition()
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
void PythonAwg::endAcquisition()
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
void PythonAwg::sleep(bool b)
{
    if (!pu_process || !pu_process->isRunning())
        return;

    QJsonObject req;
    req[QStringLiteral("method")] = QStringLiteral("sleep");
    req[QStringLiteral("sleeping")] = b;
    pu_process->sendRequest(req);
}

// ============================================================================
// readSettings()
// ============================================================================
void PythonAwg::readSettings()
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
QStringList PythonAwg::forbiddenKeys() const
{
    return {BC::Key::HW::commType, BC::Key::HW::model,
            BC::Key::PythonAwg::pythonScript, BC::Key::PythonAwg::pythonClass};
}

// ============================================================================
// parseAuxDataResult()
// ============================================================================
AuxDataStorage::AuxDataMap PythonAwg::parseAuxDataResult(const QJsonObject &response)
{
    AuxDataStorage::AuxDataMap out;
    QJsonObject result = response[QStringLiteral("result")].toObject();
    for (auto it = result.begin(); it != result.end(); ++it)
        out[it.key()] = QVariant(it.value().toDouble());
    return out;
}

#endif // BC_PYTHON_HARDWARE
