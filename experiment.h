#ifndef EXPERIMENT_H
#define EXPERIMENT_H

#include <QSharedDataPointer>
#include <QPair>
#include <QDateTime>
#include <QMetaType>
#include "ftmwconfig.h"
#include "loghandler.h"

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
    int timeDataInterval() const;
    bool isInitialized() const;
    bool isAborted() const;
    bool isDummy() const;
    FtmwConfig ftmwConfig() const;
    bool isComplete() const;
    bool hardwareSuccess() const;
    QString errorString() const;
    QMap<QString,QList<QVariant>> timeDataMap() const;
    QString startLogMessage() const;
    QString endLogMessage() const;
    LogHandler::MessageCode endLogMessageCode() const;
    QMap<QString, QPair<QVariant,QString>> headerMap() const;

    void setGasSetpoints(const QList<QPair<double,QString> > list);
    void addGasSetpoint(const double setPoint, const QString name);
    void setPressureSetpoints(const QList<QPair<double,QString> > list);
    void addPressureSetpoint(const double setPoint, const QString name);
    void setTimeDataInterval(const int t);
    void setInitialized();
    void setAborted();
    void setDummy();
    void setFtmwConfig(const FtmwConfig cfg);
    void setScopeConfig(const FtmwConfig::ScopeConfig &cfg);
    bool setFids(const QByteArray rawData);
    bool addFids(const QByteArray newData);
    void overrideTargetShots(const int target);
    void resetFids();
    void setErrorString(const QString str);
    void addTimeData(const QList<QPair<QString, QVariant> > dataList);
    void addTimeStamp();

    void setHardwareFailed();
    void incrementFtmw();

    void save();

private:
    QSharedDataPointer<ExperimentData> data;
};

class ExperimentData : public QSharedData
{
public:
    ExperimentData() : number(0), timeDataInterval(300), isInitialized(false), isAborted(false), isDummy(false), hardwareSuccess(true), endLogMessageCode(LogHandler::Highlight) {}

    int number;
    QList<QPair<double,QString> > gasSetpoints;
    QList<QPair<double,QString> > pressureSetpoints;
    QDateTime startTime;
    int timeDataInterval;
    bool isInitialized;
    bool isAborted;
    bool isDummy;
    bool hardwareSuccess;
    QString errorString;
    QString startLogMessage;
    QString endLogMessage;
    LogHandler::MessageCode endLogMessageCode;

    FtmwConfig ftmwCfg;
    QMap<QString,QList<QVariant>> timeDataMap;
};

Q_DECLARE_METATYPE(Experiment)
Q_DECLARE_TYPEINFO(Experiment, Q_MOVABLE_TYPE);

#endif // EXPERIMENT_H
