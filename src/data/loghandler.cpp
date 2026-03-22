#include <data/loghandler.h>
#include <data/storage/blackchirpcsv.h>

LogHandler::LogHandler(bool logToFile, QObject *parent) :
    QObject(parent), d_logToFile(logToFile)
{
}

LogHandler::~LogHandler()
{
}

QString LogHandler::formatForDisplay(QString text, MessageCode type, QDateTime t)
{
    QString timeStamp = t.toString();
    QString out;
    out.append(QString("<span style=\"font-size:7pt\">%1</span> ").arg(timeStamp));

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
    return out;
}

void LogHandler::logMessage(const QString text, const MessageCode type)
{
    logMessageWithTime(text,type,QDateTime::currentDateTime());
}

void LogHandler::logMessageWithTime(const QString text, const MessageCode type, QDateTime t)
{
    if(d_logToFile)
        writeToFile(text, type, t);

    if(type == Debug)
        return;  // Debug messages never appear in the UI log

    if(type == Error || type == Warning)
        emit iconUpdate(type);

    QString out = formatForDisplay(text,type,t);
    emit sendLogMessage(out);
}

void LogHandler::beginExperimentLog(int num, QString msg)
{
    d_currentExperimentNum = num;
    logMessage(msg,Highlight);
}

void LogHandler::endExperimentLog()
{
    d_currentExperimentNum = -1;
}

void LogHandler::setDebugLogging(bool enabled)
{
    d_debugLogging = enabled;
}

void LogHandler::writeToFile(const QString text, const MessageCode type, QDateTime t)
{
    QDate now = t.date();
    QString msg{text};
    msg.replace(BC::CSV::del,QString(","));
    QDir d = BlackchirpCSV::logDir();
    QString month = QString::number(now.month()).rightJustified(2,'0');
    QString yearMonth = QString::number(now.year()) + month;

    if(type == Debug)
    {
        if(!d_debugLogging)
            return;
        QFile debugLog(d.absoluteFilePath(QString("debug_") + yearMonth + ".csv"));
        if(debugLog.open(QIODevice::Append|QIODevice::Text))
        {
            QTextStream ts(&debugLog);
            if(debugLog.size() == 0)
                BlackchirpCSV::writeLine(ts,{"Timestamp","Epoch_msecs","Code","Message"});
            BlackchirpCSV::writeLine(ts,{t.toString(),t.toMSecsSinceEpoch(),
                                         QVariant::fromValue<MessageCode>(type).toString(),msg});
        }
        return;
    }

    QFile logFile(d.absoluteFilePath(yearMonth + ".csv"));
    if(logFile.open(QIODevice::Append|QIODevice::Text))
    {
        QTextStream ts(&logFile);
        if(logFile.size() == 0)
            BlackchirpCSV::writeLine(ts,{"Timestamp","Epoch_msecs","Code","Message"});
        BlackchirpCSV::writeLine(ts,{t.toString(),t.toMSecsSinceEpoch(),
                                     QVariant::fromValue<MessageCode>(type).toString(),msg});
    }

    if(d_currentExperimentNum > 0)
    {
        QDir exp = BlackchirpCSV::exptDir(d_currentExperimentNum);
        QFile expLog(exp.absoluteFilePath("log.csv"));
        if(expLog.open(QIODevice::Append|QIODevice::Text))
        {
            QTextStream ts(&expLog);
            if(expLog.size() == 0)
                BlackchirpCSV::writeLine(ts,{"Timestamp","Epoch_msecs","Code","Message"});
            BlackchirpCSV::writeLine(ts,{t.toString(),t.toMSecsSinceEpoch(),
                                         QVariant::fromValue<MessageCode>(type).toString(),msg});
        }
    }
}
