#ifndef LOGHANDLER_H
#define LOGHANDLER_H

#include <QObject>
#include <QString>
#include <QFile>

class LogHandler : public QObject
{
    Q_OBJECT
public:
    explicit LogHandler(QObject *parent = 0);
    ~LogHandler();

	//definitions of log message types
    enum MessageCode {Normal, Warning, Error, Highlight, Debug};



signals:
	//sends the formatted messages to the UI
	void sendLogMessage(const QString);
	void sendStatusMessage(const QString);

public slots:
	//access functions for transmitting messages to UI
	void logMessage(const QString text, const LogHandler::MessageCode type=LogHandler::Normal);

private:
    QFile d_logFile;
    int d_currentMonth;

    void writeToFile(const QString text, const LogHandler::MessageCode type, const QString timeStamp);
    QString makeLogFileName();


};

Q_DECLARE_METATYPE(LogHandler::MessageCode)

#endif // LOGHANDLER_H
