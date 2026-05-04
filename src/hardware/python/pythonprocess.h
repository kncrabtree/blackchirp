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
 * \brief QProcess wrapper that owns a Python hardware subprocess and
 * the JSON-lines IPC channel between Blackchirp and the user driver
 * script.
 *
 * Launches the IPC host script (\c python_hw_host.py) under the
 * configured Python interpreter, hands it the user's driver script and
 * class name, and exposes a synchronous request/response API
 * (\c sendRequest) on top of an asynchronous, line-oriented JSON
 * protocol carried on the subprocess's stdin and stdout. The Python
 * heap stays in a separate process, so a script crash cannot corrupt
 * the Qt application.
 *
 * Reentrancy: \c sendRequest() writes its request and then runs a
 * nested \c QEventLoop until either a matching response arrives, the
 * subprocess dies, or the configured timeout fires. Because the loop
 * continues to process events, relay requests, log messages, and
 * waveform pushes are handled correctly while a method call is in
 * flight.
 *
 * \sa PythonHardwareBase, CommunicationProtocol, LogHandler
 */
class PythonProcess : public QObject
{
    Q_OBJECT
public:
    explicit PythonProcess(QObject *parent = nullptr);
    ~PythonProcess();

    /*!
     * \brief Launch the Python subprocess and run the startup handshake.
     *
     * Starts \a pythonExe with \a hostScriptPath as its argument, sends
     * the initial \c _init message (carrying the hardware key, model,
     * the user script path, the class name, and the enabled proxy list
     * configured via setEnabledProxies), then sends \c initialize so
     * the user driver can populate its own state. Returns once both
     * handshake steps have completed successfully.
     *
     * \param pythonExe Python executable path (e.g.,
     *                  \c "/path/to/venv/bin/python3" or \c "python3")
     * \param hostScriptPath Absolute path to \c python_hw_host.py
     * \param userScriptPath Absolute path to the user's driver script
     * \param className Name of the Python class to instantiate inside
     *                  the user script
     * \return \c true if the process started and both handshake steps
     *         succeeded; \c false otherwise (with a processError signal
     *         emitted explaining the failure).
     */
    bool start(const QString &pythonExe, const QString &hostScriptPath, const QString &userScriptPath, const QString &className);

    /// Stop the Python subprocess if it is running. Idempotent.
    void stop();

    /// True when the subprocess is alive and accepting requests.
    bool isRunning() const;

    /*!
     * \brief Send a method call to the user driver and wait for its reply.
     *
     * Writes \a request as a JSON line on the subprocess's stdin and
     * runs a nested QEventLoop until a response with the matching id
     * arrives, the subprocess exits, or timeoutMs() elapses. Relay
     * requests, log messages, and waveform pushes received while
     * waiting are dispatched to their handlers without disturbing the
     * pending response.
     *
     * \param request JSON object with at minimum a \c "method" key;
     *                additional keys are forwarded as keyword
     *                arguments to the Python method.
     * \return Response JSON object containing either \c "result" on
     *         success or \c "error" plus \c "traceback" on failure.
     */
    QJsonObject sendRequest(const QJsonObject &request);

    /// Signature for the SettingsStorage::get bridge installed by
    /// setSettingsCallbacks().
    using SettingsGetter = std::function<QVariant(const QString &key, const QVariant &defaultVal)>;
    /// Signature for the SettingsStorage::set bridge installed by
    /// setSettingsCallbacks().
    using SettingsSetter = std::function<void(const QString &key, const QVariant &val)>;

    /*!
     * \brief Bind the CommunicationProtocol used to service \c self.comm
     * relay requests from the Python script.
     *
     * The pointer is borrowed; PythonProcess does not take ownership.
     * Trampolines call this every time their \c p_comm is reassigned.
     */
    void setComm(CommunicationProtocol *comm);

    /*!
     * \brief Bind the SettingsStorage bridge for \c self.settings relay.
     *
     * The host script's SettingsProxy turns into calls against \a getter
     * and \a setter, which the trampoline typically wires to its own
     * SettingsStorage::get and SettingsStorage::set helpers (the latter
     * is protected, so the bridge is what lets the user script update
     * persistent settings without a friend declaration).
     */
    void setSettingsCallbacks(SettingsGetter getter, SettingsSetter setter);

    /*!
     * \brief Set the hardware key and model strings forwarded to Python
     * in the \c _init message.
     *
     * The Python side exposes these as \c self.settings.key and
     * \c self.settings.model.
     */
    void setHardwareInfo(const QString &key, const QString &model);

    /// Current sendRequest() timeout in milliseconds.
    int timeoutMs() const { return d_timeoutMs; }
    /// Set the sendRequest() timeout in milliseconds.
    void setTimeoutMs(int ms) { d_timeoutMs = ms; }

    /*!
     * \brief Select which optional Python proxies are injected on the
     * next subprocess start.
     *
     * The three standard proxies (\c comm, \c settings, \c log) are
     * always injected. Optional proxies are hardware-type-specific
     * push channels, currently \c "scope" for digitizer waveform
     * push. Names not registered in the host script's factory map are
     * ignored. Must be called before start(); typically invoked from
     * the trampoline's initialize override, immediately after
     * initPythonProcess().
     */
    void setEnabledProxies(const QStringList &proxies) { d_enabledProxies = proxies; }

signals:
    /*!
     * \brief Emitted when the Python script pushes a waveform to C++.
     *
     * \param data Raw shot bytes after base64 decode, in the format
     *             configured for the active digitizer (record length
     *             × bytes per point × number of records).
     * \param shotCount Number of shots represented by \a data
     *                  (\c 1 for single-shot, \c N for pre-accumulated).
     */
    void waveformReceived(const QByteArray &data, quint64 shotCount);

    /// Emitted with a human-readable description when the subprocess
    /// fails to start, exits unexpectedly, or returns a malformed
    /// reply.
    void processError(const QString &errorString);

    /// Internal: wakes the nested QEventLoop in sendRequest() once a
    /// matching response has been parsed. Not intended for external
    /// consumers.
    void responseReady();

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
