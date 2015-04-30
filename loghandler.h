#ifndef LOGHANDLER_H
#define LOGHANDLER_H

#include <QObject>
#include <QString>
#include <QFile>

#include "datastructs.h"

class LogHandler : public QObject
{
    Q_OBJECT
public:
    explicit LogHandler(QObject *parent = 0);
    ~LogHandler();

	//definitions of log message types




signals:
	//sends the formatted messages to the UI
	void sendLogMessage(const QString);

public slots:
	//access functions for transmitting messages to UI
    void logMessage(const QString text, const BlackChirp::LogMessageCode type=BlackChirp::LogNormal);

private:
    QFile d_logFile;
    int d_currentMonth;

    void writeToFile(const QString text, const BlackChirp::LogMessageCode type, const QString timeStamp);
    QString makeLogFileName();


};

#endif // LOGHANDLER_H
