#include "loghandler.h"
#include <QDateTime>

LogHandler::LogHandler(QObject *parent) :
    QObject(parent)
{
}

void LogHandler::logMessage(const QString text, const MessageCode type)
{
	QDateTime time;
	QString out;
	out.append(QString("<span style=\"font-size:7pt\">%1</span> ").arg(time.currentDateTime().toString()));

	switch(type)
	{
	case Warning:
		out.append(QString("<span style=\"font-weight:bold\">Warning: %1</span>").arg(text));
		break;
	case Error:
		out.append(QString("<span style=\"font-weight:bold;color:red\">Error: %1</span>").arg(text));
		break;
	case Highlight:
		out.append(QString("<span style=\"font-weight:bold;color:green\">%1</span>").arg(text));
		break;
	case Normal:
	default:
		out.append(text);
		break;
	}

	//emit signal containing formatted message
	emit sendLogMessage(out);
}
