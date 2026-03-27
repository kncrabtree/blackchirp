#ifndef PYTHONPROCESS_H
#define PYTHONPROCESS_H

#ifdef BC_PYTHON_HARDWARE

#include <QObject>
#include <QProcess>
#include <QJsonObject>

#include <functional>

#include <data/loghandler.h>

class CommunicationProtocol;

/*!
 * \brief Manages a Python subprocess and JSON-lines IPC for hardware dispatch
 *
 * PythonProcess wraps QProcess to launch a Python hardware host script,
 * communicate via JSON-lines on stdin/stdout, and handle interleaved
 * relay requests (comm, settings) and log messages during method dispatch.
 */
class PythonProcess : public QObject
{
    Q_OBJECT
public:
    explicit PythonProcess(QObject *parent = nullptr);
    ~PythonProcess();

    /*!
     * \brief Launch the Python subprocess
     * \param hostScriptPath Path to python_hw_host.py
     * \param userScriptPath Path to the user's hardware script
     * \param className Name of the Python class to instantiate
     * \return True if process started and _init succeeded
     */
    bool start(const QString &hostScriptPath, const QString &userScriptPath, const QString &className);

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
     * Blocks until Python responds. While waiting, handles interleaved
     * relay requests (comm_query, get_setting, etc.) and log messages.
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

signals:
    void logMessage(const QString &msg, LogHandler::MessageCode code);
    void processError(const QString &errorString);

private:
    void handleStderr();
    QJsonObject handleRelayRequest(const QJsonObject &relayReq);
    LogHandler::MessageCode parseLogLevel(const QString &level) const;
    void writeLine(const QJsonObject &obj);
    QJsonObject readLineJson();
    QJsonObject readResponseForId(int id);

    QProcess *p_process{nullptr};
    CommunicationProtocol *p_comm{nullptr};
    SettingsGetter d_settingsGetter;
    SettingsSetter d_settingsSetter;
    QString d_hwKey;
    QString d_hwModel;
    int d_timeoutMs{30000};
    int d_nextId{1};
};

#endif // BC_PYTHON_HARDWARE
#endif // PYTHONPROCESS_H
