#include "experiment.h"
#include <QSettings>
#include <QApplication>

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

QList<QPair<double, QString> > Experiment::gasSetpoints() const
{
    return data->gasSetpoints;
}

QList<QPair<double, QString> > Experiment::pressureSetpoints() const
{
    return data->pressureSetpoints;
}

QDateTime Experiment::startTime() const
{
    return data->startTime;
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

FtmwConfig Experiment::ftmwConfig() const
{
    return data->ftmwCfg;
}

bool Experiment::isComplete() const
{
    //check each sub expriment!
    return data->ftmwCfg.isComplete();
}

bool Experiment::hardwareSuccess() const
{
    return data->hardwareSuccess;
}

QString Experiment::errorString() const
{
    return data->errorString;
}

void Experiment::setGasSetpoints(const QList<QPair<double, QString> > list)
{
    data->gasSetpoints = list;
}

void Experiment::addGasSetpoint(const double setPoint, const QString name)
{
    data->gasSetpoints.append(qMakePair(setPoint,name));
}

void Experiment::setPressureSetpoints(const QList<QPair<double, QString> > list)
{
    data->pressureSetpoints = list;
}

void Experiment::addPressureSetpoint(const double setPoint, const QString name)
{
    data->pressureSetpoints.append(qMakePair(setPoint,name));
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

    data->isInitialized = initSuccess;
    data->startTime = QDateTime::currentDateTime();

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    int num = s.value(QString("exptNum"),1).toInt();
    data->number = num;


}

void Experiment::setAborted()
{
    data->isAborted = true;
}

void Experiment::setDummy()
{
    data->isDummy = true;
}

void Experiment::setFtmwConfig(const FtmwConfig cfg)
{
    data->ftmwCfg = cfg;
}

void Experiment::setScopeConfig(const FtmwConfig::ScopeConfig &cfg)
{
    data->ftmwCfg.setScopeConfig(cfg);
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

void Experiment::setErrorString(const QString str)
{
    data->errorString = str;
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
}

