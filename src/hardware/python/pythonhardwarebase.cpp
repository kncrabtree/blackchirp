#include "pythonhardwarebase.h"

#include <QCoreApplication>
#include <QFile>
#include <QJsonObject>
#include <data/loghandler.h>

#include <hardware/core/hardwareprofilemanager.h>
#include <data/settings/hardwarekeys.h>
#include <data/bcglobals.h>

PythonHardwareBase::PythonHardwareBase(const QString &key, const QString &model)
    : d_pyKey{key}, d_pyModel{model}
{
}

PythonHardwareBase::~PythonHardwareBase()
{
    if (pu_process)
        pu_process->stop();
}

void PythonHardwareBase::stopProcess()
{
    if (pu_process)
        pu_process->stop();
}

void PythonHardwareBase::initPythonProcess(CommunicationProtocol *comm,
                                           PythonProcess::SettingsGetter getter,
                                           PythonProcess::SettingsSetter setter)
{
    pu_process = std::make_unique<PythonProcess>(nullptr);
    pu_process->setComm(comm);
    pu_process->setHardwareInfo(d_pyKey, d_pyModel);
    pu_process->setSettingsCallbacks(std::move(getter), std::move(setter));

    QObject::connect(pu_process.get(), &PythonProcess::processError,
                     [this](const QString &msg){ d_pythonErrorString = msg; });
}

bool PythonHardwareBase::testPythonConnection(CommunicationProtocol *comm)
{
    d_pythonErrorString.clear();

    if (!pu_process->isRunning()) {
        if (!startPythonProcess())
            return false;
    }

    pu_process->setComm(comm);

    QJsonObject req;
    req["method"_L1] = "test_connection"_L1;
    auto resp = pu_process->sendRequest(req);

    if (resp.contains("error"_L1)) {
        d_pythonErrorString = resp["error"_L1].toString();
        bcError(u"%1 test_connection error: %2"_s.arg(d_pyKey, d_pythonErrorString));
        return false;
    }
    return resp["result"_L1].toBool(false);
}

bool PythonHardwareBase::startPythonProcess()
{
    auto [hwType, label] = BC::Key::parseKey(d_pyKey);
    QString scriptPath = HardwareProfileManager::instance().getPythonScriptPath(hwType, label);

    if (scriptPath.isEmpty()) {
        d_pythonErrorString = "No Python script path configured"_L1;
        bcError(u"%1: %2"_s.arg(d_pyKey, d_pythonErrorString));
        return false;
    }

    QString hostScript = findHostScript();
    if (hostScript.isEmpty()) {
        d_pythonErrorString = "Cannot find python_hw_host.py"_L1;
        bcError(u"%1: %2"_s.arg(d_pyKey, d_pythonErrorString));
        return false;
    }

    QString className = HardwareProfileManager::instance().getPythonClassName(hwType, label);
    if (className.isEmpty()) {
        d_pythonErrorString = "No Python class name configured"_L1;
        bcError(u"%1: %2"_s.arg(d_pyKey, d_pythonErrorString));
        return false;
    }

    QString envPath = HardwareProfileManager::instance().getPythonEnvPath(hwType, label);
    QString pythonExe = resolvePythonExecutable(envPath);

    return pu_process->start(pythonExe, hostScript, scriptPath, className);
}

QString PythonHardwareBase::resolvePythonExecutable(const QString &envPath)
{
    if (envPath.isEmpty())
        return "python3"_L1;

    const QStringList candidates = {
        envPath + "/bin/python3"_L1,
        envPath + "/bin/python"_L1,
        envPath + "/Scripts/python.exe"_L1,
    };

    for (const auto &path : candidates) {
        if (QFile::exists(path))
            return path;
    }

    return "python3"_L1;
}

QString PythonHardwareBase::findHostScript()
{
    QStringList searchPaths = {
        QCoreApplication::applicationDirPath() + "/python_hw_host.py"_L1,
        QCoreApplication::applicationDirPath() + "/../share/blackchirp/python_hw_host.py"_L1,
    };

    for (const auto &path : searchPaths) {
        if (QFile::exists(path))
            return path;
    }

    return {};
}

void PythonHardwareBase::pythonSleep(bool b)
{
    if (!pu_process || !pu_process->isRunning())
        return;

    QJsonObject req;
    req["method"_L1] = "sleep"_L1;
    req["sleeping"_L1] = b;
    pu_process->sendRequest(req);
}

void PythonHardwareBase::pythonReadSettings()
{
    if (pu_process && pu_process->isRunning()) {
        QJsonObject req;
        req["method"_L1] = "read_settings"_L1;
        pu_process->sendRequest(req);
    }
}
