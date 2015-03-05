#ifndef EXPERIMENT_H
#define EXPERIMENT_H

#include <QSharedDataPointer>
#include <QPair>
#include <QDateTime>
#include <QMetaType>
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
    QString errorString() const;

    void setGasSetpoints(const QList<QPair<double,QString> > list);
    void addGasSetpoint(const double setPoint, const QString name);
    void setPressureSetpoints(const QList<QPair<double,QString> > list);
    void addPressureSetpoint(const double setPoint, const QString name);
    void setInitialized();
    void setAborted();
    void setDummy();
    void setFtmwConfig(const FtmwConfig cfg);
    void setScopeConfig(const FtmwConfig::ScopeConfig &cfg);
    bool setFids(const QByteArray rawData);
    bool addFids(const QByteArray newData);
    void setErrorString(const QString str);

    void setHardwareFailed();
    void incrementFtmw();

    void save();

private:
    QSharedDataPointer<ExperimentData> data;
};

class ExperimentData : public QSharedData
{
public:
    ExperimentData() : number(0), isInitialized(false), isAborted(false), isDummy(false), hardwareSuccess(true) {}

    int number;
    QList<QPair<double,QString> > gasSetpoints;
    QList<QPair<double,QString> > pressureSetpoints;
    QDateTime startTime;
    bool isInitialized;
    bool isAborted;
    bool isDummy;
    bool hardwareSuccess;
    QString errorString;

    FtmwConfig ftmwCfg;
};

Q_DECLARE_METATYPE(Experiment)
Q_DECLARE_TYPEINFO(Experiment, Q_MOVABLE_TYPE);

#endif // EXPERIMENT_H
