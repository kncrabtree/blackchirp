#ifndef LOGHANDLER_H
#define LOGHANDLER_H

#include <QObject>
#include <QString>
#include <QFile>

#include "datastructs.h"
#include "experiment.h"

class LogHandler : public QObject
{
    Q_OBJECT
public:
    explicit LogHandler(bool logToFile = true, QObject *parent = 0);
    ~LogHandler();

signals:
	//sends the formatted messages to the UI
	void sendLogMessage(const QString);
    void iconUpdate(BlackChirp::LogMessageCode);

public slots:
	//access functions for transmitting messages to UI
    void logMessage(const QString text, const BlackChirp::LogMessageCode type=BlackChirp::LogNormal);
    void logMessageWithTime(const QString text, const BlackChirp::LogMessageCode type=BlackChirp::LogNormal, QDateTime t = QDateTime::currentDateTime());
    void beginExperimentLog(const Experiment e);
    void endExperimentLog();

private:
    QFile d_logFile;
    QFile d_exptLog;
    int d_currentMonth;
    bool d_logToFile;

    void writeToFile(const QString text, const BlackChirp::LogMessageCode type, const QString timeStamp);
    QString makeLogFileName();


};

#endif // LOGHANDLER_H
