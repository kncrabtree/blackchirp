#ifndef LOGHANDLER_H
#define LOGHANDLER_H

#include <QObject>
#include <QString>
#include <QFile>
#include <QDateTime>

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

    explicit LogHandler(bool logToFile = true, QObject *parent = 0);
    ~LogHandler();

    static QString formatForDisplay(const QString text, MessageCode type, QDateTime t = QDateTime::currentDateTime());
    static QString formatForFile(const QString text, MessageCode type, QDateTime t = QDateTime::currentDateTime());

signals:
	//sends the formatted messages to the UI
    void sendLogMessage(QString);
    void iconUpdate(LogHandler::MessageCode);

public slots:
	//access functions for transmitting messages to UI
    void logMessage(const QString text, const MessageCode type=Normal);
    void logMessageWithTime(const QString text, const MessageCode type=Normal, QDateTime t = QDateTime::currentDateTime());
    void beginExperimentLog(int num, QString msg);
    void endExperimentLog();

private:
    int d_currentExperimentNum{-1};
    bool d_logToFile{true};

    void writeToFile(const QString text, const MessageCode type, QDateTime t = QDateTime::currentDateTime());
    QString makeLogFileName();


};

#endif // LOGHANDLER_H
