#ifndef LOGHANDLER_H
#define LOGHANDLER_H

#include <QObject>
#include <QString>

class LogHandler : public QObject
{
    Q_OBJECT
public:
    explicit LogHandler(QObject *parent = 0);

	//definitions of log message types
	enum MessageCode {Normal, Warning, Error, Highlight};


signals:
	//sends the formatted messages to the UI
	void sendLogMessage(const QString);
	void sendStatusMessage(const QString);

public slots:
	//access functions for transmitting messages to UI
	void logMessage(const QString text, const LogHandler::MessageCode type=LogHandler::Normal);

};

#endif // LOGHANDLER_H
