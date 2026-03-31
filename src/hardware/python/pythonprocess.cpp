#include "pythonprocess.h"

#include <QJsonDocument>
#include <QJsonObject>
#include <QCoreApplication>
#include <QElapsedTimer>
#include <QFile>

#include <hardware/core/communication/communicationprotocol.h>

PythonProcess::PythonProcess(QObject *parent) : QObject(parent)
{
}

PythonProcess::~PythonProcess()
{
    stop();
}

bool PythonProcess::start(const QString &hostScriptPath, const QString &userScriptPath, const QString &className)
{
    stop();

    if (!QFile::exists(hostScriptPath)) {
        emit processError(QString("Host script not found: %1").arg(hostScriptPath));
        return false;
    }

    if (!QFile::exists(userScriptPath)) {
        emit processError(QString("User script not found: %1").arg(userScriptPath));
        return false;
    }

    p_process = new QProcess(this);
    p_process->setProcessChannelMode(QProcess::SeparateChannels);

    connect(p_process, &QProcess::readyReadStandardError, this, &PythonProcess::handleStderr);

    p_process->start(QStringLiteral("python3"),
                     {hostScriptPath, userScriptPath, className});

    if (!p_process->waitForStarted(5000)) {
        emit processError(QString("Failed to start Python process: %1")
                              .arg(p_process->errorString()));
        delete p_process;
        p_process = nullptr;
        return false;
    }

    // Send _init message to set up proxies
    QJsonObject initReq;
    initReq[QStringLiteral("id")] = 0;
    initReq[QStringLiteral("method")] = QStringLiteral("_init");
    initReq[QStringLiteral("key")] = d_hwKey;
    initReq[QStringLiteral("model")] = d_hwModel;

    writeLine(initReq);

    // Read the _init response (with relay/log handling)
    auto resp = readResponseForId(0);
    if (resp.contains(QStringLiteral("error"))) {
        auto errMsg = QString("Python startup failed: %1").arg(
            resp[QStringLiteral("error")].toString());
        emit logMessage(errMsg, LogHandler::Error);
        emit processError(errMsg);
        stop();
        return false;
    }

    // Call the user script's initialize() so it can set up state before
    // testConnection. This mirrors the C++ HardwareObject lifecycle:
    // constructor -> initialize() -> testConnection().
    int initId = d_nextId++;
    QJsonObject userInitReq;
    userInitReq[QStringLiteral("id")] = initId;
    userInitReq[QStringLiteral("method")] = QStringLiteral("initialize");

    writeLine(userInitReq);

    auto initResp = readResponseForId(initId);
    if (initResp.contains(QStringLiteral("error"))) {
        auto errMsg = QString("Python initialize() failed: %1").arg(
            initResp[QStringLiteral("error")].toString());
        emit logMessage(errMsg, LogHandler::Error);
        emit processError(errMsg);
        stop();
        return false;
    }

    return true;
}

void PythonProcess::stop()
{
    if (!p_process)
        return;

    // Close stdin to signal EOF to the Python side
    p_process->closeWriteChannel();

    if (!p_process->waitForFinished(3000))
        p_process->kill();

    delete p_process;
    p_process = nullptr;
}

bool PythonProcess::isRunning() const
{
    return p_process && p_process->state() == QProcess::Running;
}

QJsonObject PythonProcess::sendRequest(const QJsonObject &request)
{
    if (!isRunning()) {
        QJsonObject err;
        err[QStringLiteral("error")] = QStringLiteral("Python process not running");
        return err;
    }

    int id = d_nextId++;
    QJsonObject req = request;
    req[QStringLiteral("id")] = id;

    writeLine(req);
    return readResponseForId(id);
}

void PythonProcess::setComm(CommunicationProtocol *comm)
{
    p_comm = comm;
}

void PythonProcess::setSettingsCallbacks(SettingsGetter getter, SettingsSetter setter)
{
    d_settingsGetter = std::move(getter);
    d_settingsSetter = std::move(setter);
}

void PythonProcess::setHardwareInfo(const QString &key, const QString &model)
{
    d_hwKey = key;
    d_hwModel = model;
}

// ============================================================================
// Private implementation
// ============================================================================

QJsonObject PythonProcess::readResponseForId(int id)
{
    // Read lines until we get a response with the matching id.
    // While waiting, handle relay requests and log messages.
    QElapsedTimer timer;
    timer.start();

    while (timer.elapsed() < d_timeoutMs) {
        auto line = readLineJson();
        if (line.isEmpty()) {
            // No data available yet, or process died
            if (!isRunning()) {
                // Process died — drain stderr synchronously for the error details.
                // handleStderr() won't fire because we're not in the event loop.
                QString stderrText;
                if (p_process) {
                    p_process->waitForFinished(1000);
                    QByteArray data = p_process->readAllStandardError();
                    stderrText = QString::fromUtf8(data).trimmed();
                }
                QJsonObject err;
                if (stderrText.isEmpty())
                    err[QStringLiteral("error")] = QStringLiteral("Python process terminated unexpectedly");
                else
                    err[QStringLiteral("error")] = QString("Python process terminated: %1").arg(stderrText);
                return err;
            }
            continue;
        }

        // Check for log message (unsolicited)
        if (line.contains(QStringLiteral("log"))) {
            QString msg = line[QStringLiteral("log")].toString();
            auto code = parseLogLevel(line[QStringLiteral("level")].toString());
            emit logMessage(msg, code);
            continue;
        }

        // Check for relay request from Python
        if (line.contains(QStringLiteral("relay"))) {
            auto relayResp = handleRelayRequest(line);
            writeLine(relayResp);
            continue;
        }

        // Check for matching response
        if (line.contains(QStringLiteral("id")) && line[QStringLiteral("id")].toInt() == id) {
            return line;
        }

        // Unrecognized message -- skip
    }

    // Timeout
    QJsonObject err;
    err[QStringLiteral("error")] = QString("Timeout waiting for Python response (id=%1, %2ms)")
                                       .arg(id).arg(d_timeoutMs);
    return err;
}

void PythonProcess::handleStderr()
{
    if (!p_process)
        return;

    QByteArray data = p_process->readAllStandardError();
    if (!data.isEmpty()) {
        QString text = QString::fromUtf8(data).trimmed();
        if (!text.isEmpty())
            emit logMessage(QString("Python stderr: %1").arg(text), LogHandler::Warning);
    }
}

QJsonObject PythonProcess::handleRelayRequest(const QJsonObject &relayReq)
{
    QString relayType = relayReq[QStringLiteral("relay")].toString();
    QJsonObject resp;

    if (relayType == QLatin1String("comm_query")) {
        if (!p_comm) {
            resp[QStringLiteral("relay_error")] = QStringLiteral("No communication protocol available");
            return resp;
        }
        QString cmd = relayReq[QStringLiteral("cmd")].toString();
        QByteArray result = p_comm->queryCmd(cmd);
        resp[QStringLiteral("relay_result")] = QString::fromUtf8(result);

    } else if (relayType == QLatin1String("comm_write")) {
        if (!p_comm) {
            resp[QStringLiteral("relay_error")] = QStringLiteral("No communication protocol available");
            return resp;
        }
        QString cmd = relayReq[QStringLiteral("cmd")].toString();
        bool ok = p_comm->writeCmd(cmd);
        resp[QStringLiteral("relay_result")] = ok;

    } else if (relayType == QLatin1String("comm_read_bytes")) {
        if (!p_comm) {
            resp[QStringLiteral("relay_error")] = QStringLiteral("No communication protocol available");
            return resp;
        }
        qint64 n = relayReq[QStringLiteral("n")].toInteger();
        QByteArray data = p_comm->readBytes(n);
        resp[QStringLiteral("relay_result")] = QString::fromLatin1(data.toBase64());

    } else if (relayType == QLatin1String("comm_write_binary")) {
        if (!p_comm) {
            resp[QStringLiteral("relay_error")] = QStringLiteral("No communication protocol available");
            return resp;
        }
        QByteArray data = QByteArray::fromBase64(
            relayReq[QStringLiteral("data")].toString().toLatin1());
        bool ok = p_comm->writeBinary(data);
        resp[QStringLiteral("relay_result")] = ok;

    } else if (relayType == QLatin1String("get_setting")) {
        if (!d_settingsGetter) {
            resp[QStringLiteral("relay_error")] = QStringLiteral("No settings getter available");
            return resp;
        }
        QString key = relayReq[QStringLiteral("key")].toString();
        QVariant defaultVal = relayReq[QStringLiteral("default")].toVariant();
        QVariant val = d_settingsGetter(key, defaultVal);
        resp[QStringLiteral("relay_result")] = QJsonValue::fromVariant(val);

    } else if (relayType == QLatin1String("set_setting")) {
        if (!d_settingsSetter) {
            resp[QStringLiteral("relay_error")] = QStringLiteral("No settings setter available");
            return resp;
        }
        QString key = relayReq[QStringLiteral("key")].toString();
        QVariant val = relayReq[QStringLiteral("value")].toVariant();
        d_settingsSetter(key, val);
        resp[QStringLiteral("relay_result")] = QJsonValue();

    } else {
        resp[QStringLiteral("relay_error")] = QString("Unknown relay type: %1").arg(relayType);
    }

    return resp;
}

LogHandler::MessageCode PythonProcess::parseLogLevel(const QString &level) const
{
    if (level == QLatin1String("Warning"))
        return LogHandler::Warning;
    if (level == QLatin1String("Error"))
        return LogHandler::Error;
    if (level == QLatin1String("Highlight"))
        return LogHandler::Highlight;
    if (level == QLatin1String("Debug"))
        return LogHandler::Debug;
    return LogHandler::Normal;
}

void PythonProcess::writeLine(const QJsonObject &obj)
{
    if (!p_process || p_process->state() != QProcess::Running)
        return;

    QByteArray line = QJsonDocument(obj).toJson(QJsonDocument::Compact);
    line.append('\n');
    p_process->write(line);
    p_process->waitForBytesWritten(1000);
}

QJsonObject PythonProcess::readLineJson()
{
    if (!p_process)
        return {};

    // Wait for data to be available
    if (!p_process->canReadLine()) {
        if (!p_process->waitForReadyRead(100))
            return {};
    }

    if (!p_process->canReadLine())
        return {};

    QByteArray line = p_process->readLine();
    if (line.isEmpty())
        return {};

    QJsonParseError err;
    QJsonDocument doc = QJsonDocument::fromJson(line.trimmed(), &err);
    if (err.error != QJsonParseError::NoError || !doc.isObject())
        return {};

    return doc.object();
}
