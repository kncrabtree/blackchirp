#include "experiment.h"

class ExperimentData : public QSharedData
{
public:
    ExperimentData() : number(0), isInitialized(false), isAborted(false), isDummy(false) {}

    int number;
    QList<QPair<double,QString> > gasSetpoints;
    QList<QPair<double,QString> > pressureSetpoints;
    QDateTime startTime;
    bool isInitialized;
    bool isAborted;
    bool isDummy;

    FtmwConfig ftmwCfg;
};

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
    data->isInitialized = true;
    data->startTime = QDateTime::currentDateTime();
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

void Experiment::incrementFtmw()
{
    data->ftmwCfg.increment();
}

