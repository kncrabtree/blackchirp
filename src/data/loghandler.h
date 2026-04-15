#ifndef LOGHANDLER_H
#define LOGHANDLER_H

#include <atomic>

#include <QObject>
#include <QString>
#include <QDateTime>
#include <QMutex>
#include <QAnyStringView>

using namespace Qt::Literals::StringLiterals;

class LogHandler : public QObject
{
    Q_OBJECT
public:
    enum MessageCode {
        Normal,
        Warning,
        Error,
        Highlight,
        Debug
    };
    Q_ENUM(MessageCode)

    explicit LogHandler(bool logToFile = true, QObject *parent = nullptr);
    ~LogHandler();

    static LogHandler &instance();

    // Primary API: accepts QString, string literals, QStringView, etc. without conversion
    void log(QAnyStringView text, MessageCode type = Normal);

    static QString formatForDisplay(const QString &text, MessageCode type,
                                    QDateTime t = QDateTime::currentDateTime());

signals:
    void sendLogMessage(QString);
    void iconUpdate(LogHandler::MessageCode);

public slots:
    // Shim: forwards to log() so existing emit logMessage() call sites keep compiling.
    void logMessage(const QString text, const MessageCode type = Normal);
    void logMessageWithTime(const QString text, const MessageCode type = Normal,
                            QDateTime t = QDateTime::currentDateTime());
    void beginExperimentLog(int num, const QString &msg);
    void endExperimentLog();
    void setDebugLogging(bool enabled);

private:
    static LogHandler *s_instance;

    std::atomic<int>  d_currentExperimentNum{-1};
    std::atomic<bool> d_logToFile{true};
    std::atomic<bool> d_debugLogging{false};
    QMutex d_fileMutex;

    void doLog(const QString &text, MessageCode type, const QDateTime &t);
    void writeToFile(const QString &text, MessageCode type, const QDateTime &t);
};

// Convenience free functions — callable from any context, any thread
void bcLog(QAnyStringView text, LogHandler::MessageCode type = LogHandler::Normal);
void bcWarn(QAnyStringView text);
void bcError(QAnyStringView text);
void bcDebug(QAnyStringView text);
void bcHighlight(QAnyStringView text);

#endif // LOGHANDLER_H
