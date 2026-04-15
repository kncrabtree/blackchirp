#include <data/loghandler.h>
#include <data/storage/blackchirpcsv.h>

#include <QFile>
#include <QMutexLocker>

LogHandler *LogHandler::s_instance = nullptr;

LogHandler &LogHandler::instance()
{
    if (!s_instance)
        s_instance = new LogHandler;
    return *s_instance;
}

LogHandler::LogHandler(bool logToFile, QObject *parent)
    : QObject(parent), d_logToFile{logToFile}
{
}

LogHandler::~LogHandler()
{
}

QString LogHandler::formatForDisplay(const QString &text, MessageCode type, QDateTime t)
{
    QString timeStamp = t.toString();
    QString out;
    out.append(u"<span style=\"font-size:7pt\">%1</span> "_s.arg(timeStamp));

    switch(type)
    {
    case Warning:
        out.append(u"<span style=\"font-weight:bold\">Warning: %1</span>"_s.arg(text));
        break;
    case Error:
        out.append(u"<span style=\"font-weight:bold;color:red\">Error: %1</span>"_s.arg(text));
        break;
    case Highlight:
        out.append(u"<span style=\"font-weight:bold;color:green\">%1</span>"_s.arg(text));
        break;
    case Normal:
    case Debug:
    default:
        out.append(text);
        break;
    }
    return out;
}

void LogHandler::log(QAnyStringView text, MessageCode type)
{
    doLog(text.toString(), type, QDateTime::currentDateTime());
}

void LogHandler::logMessage(const QString text, const MessageCode type)
{
    doLog(text, type, QDateTime::currentDateTime());
}

void LogHandler::logMessageWithTime(const QString text, const MessageCode type, QDateTime t)
{
    doLog(text, type, t);
}

void LogHandler::doLog(const QString &text, MessageCode type, const QDateTime &t)
{
    if (d_logToFile)
        writeToFile(text, type, t);

    if (type == Debug)
        return;

    if (type == Error || type == Warning)
        emit iconUpdate(type);

    emit sendLogMessage(formatForDisplay(text, type, t));
}

void LogHandler::beginExperimentLog(int num, const QString &msg)
{
    d_currentExperimentNum = num;
    doLog(msg, Highlight, QDateTime::currentDateTime());
}

void LogHandler::endExperimentLog()
{
    d_currentExperimentNum = -1;
}

void LogHandler::setDebugLogging(bool enabled)
{
    d_debugLogging = enabled;
}

void LogHandler::writeToFile(const QString &text, MessageCode type, const QDateTime &t)
{
    QString msg{text};
    msg.replace(BC::CSV::del, ","_L1);

    QMutexLocker locker(&d_fileMutex);

    QDir d = BlackchirpCSV::logDir();
    QDate now = t.date();
    QString month = QString::number(now.month()).rightJustified(2, '0');
    QString yearMonth = QString::number(now.year()) + month;

    if (type == Debug)
    {
        if (!d_debugLogging)
            return;
        QFile debugLog(d.absoluteFilePath(u"debug_"_s + yearMonth + u".csv"_s));
        if (debugLog.open(QIODevice::Append | QIODevice::Text))
        {
            QTextStream ts(&debugLog);
            if (debugLog.size() == 0)
                BlackchirpCSV::writeLine(ts, {u"Timestamp"_s, u"Epoch_msecs"_s, u"Code"_s, u"Message"_s});
            BlackchirpCSV::writeLine(ts, {t.toString(), t.toMSecsSinceEpoch(),
                                          QVariant::fromValue<MessageCode>(type).toString(), msg});
        }
        return;
    }

    QFile logFile(d.absoluteFilePath(yearMonth + u".csv"_s));
    if (logFile.open(QIODevice::Append | QIODevice::Text))
    {
        QTextStream ts(&logFile);
        if (logFile.size() == 0)
            BlackchirpCSV::writeLine(ts, {u"Timestamp"_s, u"Epoch_msecs"_s, u"Code"_s, u"Message"_s});
        BlackchirpCSV::writeLine(ts, {t.toString(), t.toMSecsSinceEpoch(),
                                      QVariant::fromValue<MessageCode>(type).toString(), msg});
    }

    int experimentNum = d_currentExperimentNum;
    if (experimentNum > 0)
    {
        QDir exp = BlackchirpCSV::exptDir(experimentNum);
        QFile expLog(exp.absoluteFilePath(u"log.csv"_s));
        if (expLog.open(QIODevice::Append | QIODevice::Text))
        {
            QTextStream ts(&expLog);
            if (expLog.size() == 0)
                BlackchirpCSV::writeLine(ts, {u"Timestamp"_s, u"Epoch_msecs"_s, u"Code"_s, u"Message"_s});
            BlackchirpCSV::writeLine(ts, {t.toString(), t.toMSecsSinceEpoch(),
                                          QVariant::fromValue<MessageCode>(type).toString(), msg});
        }
    }
}

void bcLog(QAnyStringView text, LogHandler::MessageCode type)
{
    LogHandler::instance().log(text, type);
}

void bcWarn(QAnyStringView text)
{
    LogHandler::instance().log(text, LogHandler::Warning);
}

void bcError(QAnyStringView text)
{
    LogHandler::instance().log(text, LogHandler::Error);
}

void bcDebug(QAnyStringView text)
{
    LogHandler::instance().log(text, LogHandler::Debug);
}

void bcHighlight(QAnyStringView text)
{
    LogHandler::instance().log(text, LogHandler::Highlight);
}
