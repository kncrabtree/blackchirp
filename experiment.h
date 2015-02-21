#ifndef EXPERIMENT_H
#define EXPERIMENT_H

#include <QSharedDataPointer>
#include <QPair>
#include <QDateTime>
#include "ftmwconfig.h"

class ExperimentData;

class Experiment
{
public:
    Experiment();
    Experiment(const Experiment &);
    Experiment &operator=(const Experiment &);
    ~Experiment();

    int number() const;
    QList<QPair<double,QString> > gasSetpoints() const;
    QList<QPair<double,QString> > pressureSetpoints() const;
    QDateTime startTime() const;
    bool isInitialized() const;
    bool isAborted() const;
    bool isDummy() const;
    FtmwConfig ftmwConfig() const;
    bool isComplete() const;
    bool hardwareSuccess() const;

    void setGasSetpoints(const QList<QPair<double,QString> > list);
    void addGasSetpoint(const double setPoint, const QString name);
    void setPressureSetpoints(const QList<QPair<double,QString> > list);
    void addPressureSetpoint(const double setPoint, const QString name);
    void setInitialized();
    void setAborted();
    void setDummy();
    void setFtmwConfig(const FtmwConfig cfg);
    void setScopeConfig(const FtmwConfig::ScopeConfig &cfg);

    void setHardwareFailed();
    void incrementFtmw();

private:
    QSharedDataPointer<ExperimentData> data;
};

#endif // EXPERIMENT_H
