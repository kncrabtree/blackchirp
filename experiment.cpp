#include "experiment.h"

#include <QSettings>
#include <QApplication>
#include <QDir>
#include <QFile>
#include <QTextStream>

Experiment::Experiment() : data(new ExperimentData)
{

}

Experiment::Experiment(const Experiment &rhs) : data(rhs.data)
{

}

Experiment &Experiment::operator=(const Experiment &rhs)
{
    if (this != &rhs)
        data.operator=(rhs.data);
    return *this;
}

Experiment::~Experiment()
{

}

int Experiment::number() const
{
    return data->number;
}

QDateTime Experiment::startTime() const
{
    return data->startTime;
}

int Experiment::timeDataInterval() const
{
    return data->timeDataInterval;
}

bool Experiment::isInitialized() const
{
    return data->isInitialized;
}

bool Experiment::isAborted() const
{
    return data->isAborted;
}

bool Experiment::isDummy() const
{
    return data->isDummy;
}

bool Experiment::isLifWaiting() const
{
    return data->waitForLifSet;
}

FtmwConfig Experiment::ftmwConfig() const
{
    return data->ftmwCfg;
}

PulseGenConfig Experiment::pGenConfig() const
{
    return data->pGenCfg;
}

FlowConfig Experiment::flowConfig() const
{
    return data->flowCfg;
}

LifConfig Experiment::lifConfig() const
{
    return data->lifCfg;
}

bool Experiment::isComplete() const
{
    //check each sub expriment!
    return (data->ftmwCfg.isComplete() && data->lifCfg.isComplete());
}

bool Experiment::hardwareSuccess() const
{
    return data->hardwareSuccess;
}

QString Experiment::errorString() const
{
    return data->errorString;
}

QMap<QString, QList<QVariant> > Experiment::timeDataMap() const
{
    return data->timeDataMap;
}

QString Experiment::startLogMessage() const
{
    return data->startLogMessage;
}

QString Experiment::endLogMessage() const
{
    return data->endLogMessage;
}

BlackChirp::LogMessageCode Experiment::endLogMessageCode() const
{
    return data->endLogMessageCode;
}

QMap<QString, QPair<QVariant, QString> > Experiment::headerMap() const
{
    QMap<QString, QPair<QVariant, QString> > out;

    out.insert(QString("AuxDataInterval"),qMakePair(data->timeDataInterval,QString("s")));

    if(data->ftmwCfg.isEnabled())
        out.unite(data->ftmwCfg.headerMap());

    if(data->lifCfg.isEnabled())
        out.unite(data->lifCfg.headerMap());

    out.unite(data->pGenCfg.headerMap());
    out.unite(data->flowCfg.headerMap());

    return out;
}

void Experiment::setTimeDataInterval(const int t)
{
    data->timeDataInterval = t;
}

void Experiment::setInitialized()
{
    bool initSuccess = true;
    if(ftmwConfig().isEnabled())
    {
        initSuccess = data->ftmwCfg.prepareForAcquisition();
        if(!initSuccess)
            data->errorString = data->ftmwCfg.errorString();
    }



    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    int num = s.value(QString("exptNum"),0).toInt()+1;
    data->number = num;

    if(ftmwConfig().isEnabled() && ftmwConfig().type() == BlackChirp::FtmwPeakUp)
    {
        data->startLogMessage = QString("Peak up mode started.");
        data->endLogMessage = QString("Peak up mode ended.");
    }
    else
    {
        data->startLogMessage = QString("Starting experiment %1.").arg(num);
        data->endLogMessage = QString("Experiment %1 complete.").arg(num);
    }

    QDir d(BlackChirp::getExptDir(num));
    if(!d.exists())
    {
        initSuccess = d.mkpath(d.absolutePath());
        if(!initSuccess)
        {
            data->isInitialized = false;
            data->errorString = QString("Could not create the directory %1 for saving.").arg(d.absolutePath());
            return;
        }
    }

    //write headers; chirps, etc
    //scan header
    QFile hdr(BlackChirp::getExptFile(num,BlackChirp::HeaderFile));
    if(!hdr.open(QIODevice::WriteOnly))
    {
        data->isInitialized = false;
        data->errorString = QString("Could not open the file %1 for writing.").arg(hdr.fileName());
        return;
    }
    else
    {
        QTextStream t(&hdr);
        t << BlackChirp::headerMapToString(headerMap());
        t.flush();
        hdr.close();
    }

    //chirp file
    if(data->ftmwCfg.isEnabled())
    {
        QFile chp(BlackChirp::getExptFile(num,BlackChirp::ChirpFile));
        if(!chp.open(QIODevice::WriteOnly))
        {
            data->isInitialized = false;
            data->errorString = QString("Could not open the file %1 for writing.").arg(chp.fileName());
            return;
        }
        else
        {
            QTextStream t(&chp);
            t << data->ftmwCfg.chirpConfig().toString();
            t.flush();
            chp.close();
        }
    }


    data->isInitialized = initSuccess;
    data->startTime = QDateTime::currentDateTime();

    if(initSuccess)
        s.setValue(QString("exptNum"),0); //FIXME

}

void Experiment::setAborted()
{
    data->isAborted = true;
    if(ftmwConfig().isEnabled() && (ftmwConfig().type() == BlackChirp::FtmwTargetShots || ftmwConfig().type() == BlackChirp::FtmwTargetTime ))
    {
        data->endLogMessage = QString("Experiment %1 aborted.").arg(number());
        data->endLogMessageCode = BlackChirp::LogError;
    }
    else if(ftmwConfig().isEnabled() && lifConfig().isEnabled() && !lifConfig().isComplete())
    {
        data->endLogMessage = QString("Experiment %1 aborted.").arg(number());
        data->endLogMessageCode = BlackChirp::LogError;
    }
}

void Experiment::setDummy()
{
    data->isDummy = true;
}

void Experiment::setLifWaiting(bool wait)
{
    data->waitForLifSet = wait;
}

void Experiment::setFtmwConfig(const FtmwConfig cfg)
{
    data->ftmwCfg = cfg;
}

void Experiment::setScopeConfig(const BlackChirp::FtmwScopeConfig &cfg)
{
    data->ftmwCfg.setScopeConfig(cfg);
}

void Experiment::setLifConfig(const LifConfig cfg)
{
    data->lifCfg = cfg;
}


bool Experiment::setFids(const QByteArray rawData)
{
    if(!data->ftmwCfg.setFids(rawData))
    {
        setErrorString(ftmwConfig().errorString());
        return false;
    }

    return true;
}

bool Experiment::addFids(const QByteArray newData)
{
    if(!data->ftmwCfg.addFids(newData))
    {
        setErrorString(ftmwConfig().errorString());
        return false;
    }

    return true;
}

bool Experiment::addLifWaveform(const LifTrace t)
{
    return data->lifCfg.addWaveform(t);
}

void Experiment::overrideTargetShots(const int target)
{
    data->ftmwCfg.setTargetShots(target);
}

void Experiment::resetFids()
{
    data->ftmwCfg.resetFids();
}

void Experiment::setPulseGenConfig(const PulseGenConfig c)
{
    data->pGenCfg = c;
}

void Experiment::setFlowConfig(const FlowConfig c)
{
    data->flowCfg = c;
}

void Experiment::setErrorString(const QString str)
{
    data->errorString = str;
}

void Experiment::addTimeData(const QList<QPair<QString, QVariant> > dataList)
{
    for(int i=0; i<dataList.size(); i++)
    {
        QString key = dataList.at(i).first;
        QVariant value = dataList.at(i).second;

        if(data->timeDataMap.contains(key))
            data->timeDataMap[key].append(value);
        else
        {
            QList<QVariant> newList;
            newList.append(value);
            data->timeDataMap.insert(key,newList);
        }
    }
}

void Experiment::addTimeStamp()
{
    QString key("exptTimeStamp");
    if(data->timeDataMap.contains(key))
        data->timeDataMap[key].append(QDateTime::currentDateTime());
    else
    {
        QList<QVariant> newList;
        newList.append(QDateTime::currentDateTime());
        data->timeDataMap.insert(key,newList);
    }
}

void Experiment::setHardwareFailed()
{
    data->hardwareSuccess = false;
}

void Experiment::incrementFtmw()
{
    data->ftmwCfg.increment();
}

void Experiment::save()
{
    //record validation keys
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    QString keys = s.value(QString("knownValidationKeys"),QString("")).toString();
    QStringList knownKeyList = keys.split(QChar(';'),QString::SkipEmptyParts);

    auto it = data->timeDataMap.constBegin();
    while(it != data->timeDataMap.constEnd())
    {
        QString key = it.key();
        if(!knownKeyList.contains(key))
            knownKeyList.append(key);
        it++;
    }

    keys.clear();
    if(knownKeyList.size() > 0)
    {
        keys = knownKeyList.at(0);
        for(int i=1; i<knownKeyList.size();i++)
            keys.append(QString(";%1").arg(knownKeyList.at(i)));

        s.setValue(QString("knownValidationKeys"),keys);
    }

    //rewrite header file
    QFile hdr(BlackChirp::getExptFile(data->number,BlackChirp::HeaderFile));
    if(hdr.open(QIODevice::WriteOnly))
    {
        QTextStream t(&hdr);
        t << BlackChirp::headerMapToString(headerMap());
        t.flush();
        hdr.close();
    }

    //write fid (NOTE: this code is tested, and it works.
    //Don't want to waste disk space with useless FIDs
    if(ftmwConfig().isEnabled())
        ftmwConfig().writeFidFile(data->number);

//    if(lifConfig().isEnabled())
//        lifConfig().writeLifFile(data->number);
}

