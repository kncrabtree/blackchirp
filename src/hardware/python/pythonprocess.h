#ifndef PYTHONPROCESS_H
#define PYTHONPROCESS_H

#include <QByteArray>
#include <QJsonObject>
#include <QObject>
#include <QProcess>
#include <QStringList>

#include <functional>

#include <data/loghandler.h>

class CommunicationProtocol;

/*!
 * \brief Manages a Python subprocess and JSON-lines IPC for hardware dispatch
 *
 * PythonProcess wraps QProcess to launch a Python hardware host script,
 * communicate via JSON-lines on stdin/stdout, and handle interleaved
 * relay requests (comm, settings) and log messages during method dispatch.
 *
 * Reads are event-driven via readyReadStandardOutput connected to onReadyRead().
 * sendRequest() uses a nested QEventLoop to wait for a response while still
 * processing events (relay requests, log messages, waveform pushes).
 */
class PythonProcess : public QObject
{
    Q_OBJECT
public:
    explicit PythonProcess(QObject *parent = nullptr);
    ~PythonProcess();

    /*!
     * \brief Launch the Python subprocess
     * \param pythonExe Python executable path (e.g. "/path/to/venv/bin/python3" or "python3")
     * \param hostScriptPath Path to python_hw_host.py
     * \param userScriptPath Path to the user's hardware script
     * \param className Name of the Python class to instantiate
     * \return True if process started and _init succeeded
     */
    bool start(const QString &pythonExe, const QString &hostScriptPath, const QString &userScriptPath, const QString &className);

    /*!
     * \brief Stop the Python subprocess
     */
    void stop();

    /*!
     * \brief Check if the subprocess is running
     */
    bool isRunning() const;

    /*!
     * \brief Send a method call to the Python subprocess
     *
     * Writes the request to stdin, then enters a nested QEventLoop to wait
     * for the response. While waiting, onReadyRead() handles interleaved
     * relay requests, log messages, and waveform pushes.
     *
     * \param request JSON object with "method" and optional parameters
     * \return Response JSON object with "result" or "error"
     */
    QJsonObject sendRequest(const QJsonObject &request);

    using SettingsGetter = std::function<QVariant(const QString &key, const QVariant &defaultVal)>;
    using SettingsSetter = std::function<void(const QString &key, const QVariant &val)>;

    void setComm(CommunicationProtocol *comm);
    void setSettingsCallbacks(SettingsGetter getter, SettingsSetter setter);
    void setHardwareInfo(const QString &key, const QString &model);

    int timeoutMs() const { return d_timeoutMs; }
    void setTimeoutMs(int ms) { d_timeoutMs = ms; }

    void setEnabledProxies(const QStringList &proxies) { d_enabledProxies = proxies; }

signals:
    void waveformReceived(const QByteArray &data, quint64 shotCount);
    void processError(const QString &errorString);
    void responseReady();  // internal: wakes sendRequest() event loop

private slots:
    void onReadyRead();
    void handleStderr();

private:
    QJsonObject handleRelayRequest(const QJsonObject &relayReq);
    LogHandler::MessageCode parseLogLevel(const QString &level) const;
    void writeLine(const QJsonObject &obj);

    QProcess *p_process{nullptr};
    CommunicationProtocol *p_comm{nullptr};
    SettingsGetter d_settingsGetter;
    SettingsSetter d_settingsSetter;
    QString d_hwKey;
    QString d_hwModel;
    int d_timeoutMs{30000};
    int d_nextId{1};
    QStringList d_enabledProxies;

    // State for event-driven read loop
    QByteArray d_readBuf;
    bool d_waitingForResponse{false};
    int d_expectedId{-1};
    QJsonObject d_pendingResponse;
};

#endif // PYTHONPROCESS_H
