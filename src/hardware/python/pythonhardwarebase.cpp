#include "pythonhardwarebase.h"

#include <QCoreApplication>
#include <QFile>
#include <QJsonObject>
#include <QDebug>

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

QStringList PythonHardwareBase::pythonForbiddenKeys()
{
    return {BC::Key::HW::commType, BC::Key::HW::model};
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
}

bool PythonHardwareBase::testPythonConnection(CommunicationProtocol *comm)
{
    if (!pu_process->isRunning()) {
        if (!startPythonProcess())
            return false;
    }

    pu_process->setComm(comm);

    QJsonObject req;
    req[QStringLiteral("method")] = QStringLiteral("test_connection");
    auto resp = pu_process->sendRequest(req);

    if (resp.contains(QStringLiteral("error"))) {
        d_pythonErrorString = resp[QStringLiteral("error")].toString();
        qDebug() << d_pyKey << "test_connection error:" << d_pythonErrorString;
        return false;
    }
    return resp[QStringLiteral("result")].toBool(false);
}

bool PythonHardwareBase::startPythonProcess()
{
    auto [hwType, label] = BC::Key::parseKey(d_pyKey);
    QString scriptPath = HardwareProfileManager::instance().getPythonScriptPath(hwType, label);

    if (scriptPath.isEmpty()) {
        d_pythonErrorString = QStringLiteral("No Python script path configured");
        qDebug() << d_pyKey << d_pythonErrorString;
        return false;
    }

    QString hostScript = findHostScript();
    if (hostScript.isEmpty()) {
        d_pythonErrorString = QStringLiteral("Cannot find python_hw_host.py");
        qDebug() << d_pyKey << d_pythonErrorString;
        return false;
    }

    QString className = HardwareProfileManager::instance().getPythonClassName(hwType, label);
    if (className.isEmpty()) {
        d_pythonErrorString = QStringLiteral("No Python class name configured");
        qDebug() << d_pyKey << d_pythonErrorString;
        return false;
    }

    QString envPath = HardwareProfileManager::instance().getPythonEnvPath(hwType, label);
    QString pythonExe = resolvePythonExecutable(envPath);

    return pu_process->start(pythonExe, hostScript, scriptPath, className);
}

QString PythonHardwareBase::resolvePythonExecutable(const QString &envPath)
{
    if (envPath.isEmpty())
        return QStringLiteral("python3");

    const QStringList candidates = {
        envPath + QStringLiteral("/bin/python3"),
        envPath + QStringLiteral("/bin/python"),
        envPath + QStringLiteral("/Scripts/python.exe"),
    };

    for (const auto &path : candidates) {
        if (QFile::exists(path))
            return path;
    }

    return QStringLiteral("python3");
}

QString PythonHardwareBase::findHostScript()
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

void PythonHardwareBase::pythonSleep(bool b)
{
    if (!pu_process || !pu_process->isRunning())
        return;

    QJsonObject req;
    req[QStringLiteral("method")] = QStringLiteral("sleep");
    req[QStringLiteral("sleeping")] = b;
    pu_process->sendRequest(req);
}

void PythonHardwareBase::pythonReadSettings()
{
    if (pu_process && pu_process->isRunning()) {
        QJsonObject req;
        req[QStringLiteral("method")] = QStringLiteral("read_settings");
        pu_process->sendRequest(req);
    }
}
