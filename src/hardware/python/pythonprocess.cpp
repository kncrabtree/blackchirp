#include "pythonprocess.h"

#include <QEventLoop>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QFile>
#include <QTimer>

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

    connect(p_process, &QProcess::readyReadStandardOutput, this, &PythonProcess::onReadyRead);
    connect(p_process, &QProcess::readyReadStandardError,  this, &PythonProcess::handleStderr);

    p_process->start(QStringLiteral("python3"),
                     {hostScriptPath, userScriptPath, className});

    if (!p_process->waitForStarted(5000)) {
        emit processError(QString("Failed to start Python process: %1")
                              .arg(p_process->errorString()));
        delete p_process;
        p_process = nullptr;
        return false;
    }

    // Send _init message to inject proxies
    QJsonObject initReq;
    initReq[QStringLiteral("method")]  = QStringLiteral("_init");
    initReq[QStringLiteral("key")]     = d_hwKey;
    initReq[QStringLiteral("model")]   = d_hwModel;
    initReq[QStringLiteral("proxies")] = QJsonArray::fromStringList(d_enabledProxies);

    auto resp = sendRequest(initReq);
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
    QJsonObject userInitReq;
    userInitReq[QStringLiteral("method")] = QStringLiteral("initialize");

    auto initResp = sendRequest(userInitReq);
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

    d_readBuf.clear();
    d_waitingForResponse = false;
    d_pendingResponse = {};
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

    d_expectedId = id;
    d_waitingForResponse = true;
    d_pendingResponse = {};

    writeLine(req);

    QEventLoop loop;
    connect(this, &PythonProcess::responseReady, &loop, &QEventLoop::quit);
    connect(p_process, &QProcess::finished, &loop, [this, &loop]() {
        onReadyRead(); // drain any remaining stdout before giving up
        loop.quit();
    });
    QTimer::singleShot(d_timeoutMs, &loop, &QEventLoop::quit);
    loop.exec();

    d_waitingForResponse = false;

    if (d_pendingResponse.isEmpty()) {
        QJsonObject err;
        if (!isRunning()) {
            QString stderrText;
            if (p_process) {
                QByteArray data = p_process->readAllStandardError();
                stderrText = QString::fromUtf8(data).trimmed();
            }
            if (stderrText.isEmpty())
                err[QStringLiteral("error")] = QStringLiteral("Python process terminated unexpectedly");
            else
                err[QStringLiteral("error")] = QString("Python process terminated: %1").arg(stderrText);
        } else {
            err[QStringLiteral("error")] = QString("Timeout waiting for Python response (id=%1, %2ms)")
                                               .arg(id).arg(d_timeoutMs);
        }
        return err;
    }

    return d_pendingResponse;
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
// Private slots
// ============================================================================

void PythonProcess::onReadyRead()
{
    if (!p_process)
        return;

    d_readBuf.append(p_process->readAllStandardOutput());

    while (true) {
        int idx = d_readBuf.indexOf('\n');
        if (idx < 0)
            break;

        QByteArray line = d_readBuf.left(idx).trimmed();
        d_readBuf.remove(0, idx + 1);

        if (line.isEmpty())
            continue;

        QJsonParseError parseErr;
        QJsonDocument doc = QJsonDocument::fromJson(line, &parseErr);
        if (parseErr.error != QJsonParseError::NoError || !doc.isObject())
            continue;

        QJsonObject msg = doc.object();

        // Log message (unsolicited)
        if (msg.contains(QStringLiteral("log"))) {
            QString text = msg[QStringLiteral("log")].toString();
            auto code = parseLogLevel(msg[QStringLiteral("level")].toString());
            emit logMessage(text, code);
            continue;
        }

        // Waveform push (unsolicited)
        if (msg.contains(QStringLiteral("waveform"))) {
            QByteArray data = QByteArray::fromBase64(
                msg[QStringLiteral("waveform")].toString().toLatin1());
            auto shots = static_cast<quint64>(msg[QStringLiteral("shots")].toInteger(1));
            emit waveformReceived(data, shots);
            continue;
        }

        // Relay request from Python
        if (msg.contains(QStringLiteral("relay"))) {
            auto relayResp = handleRelayRequest(msg);
            writeLine(relayResp);
            continue;
        }

        // Response to sendRequest
        if (msg.contains(QStringLiteral("id"))) {
            int msgId = msg[QStringLiteral("id")].toInt();
            if (d_waitingForResponse && msgId == d_expectedId) {
                d_pendingResponse = msg;
                emit responseReady();
            }
            continue;
        }
    }
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

// ============================================================================
// Private helpers
// ============================================================================

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
