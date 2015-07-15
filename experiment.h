#ifndef EXPERIMENT_H
#define EXPERIMENT_H

#include <QSharedDataPointer>
#include <QPair>
#include <QDateTime>
#include <QMetaType>

#include "ftmwconfig.h"
#include "datastructs.h"
#include "pulsegenconfig.h"
#include "flowconfig.h"
#include "lifconfig.h"

class ExperimentData;

class Experiment
{
public:
    Experiment();
    Experiment(const Experiment &);
    Experiment &operator=(const Experiment &);
    ~Experiment();

    int number() const;
    QDateTime startTime() const;
    int timeDataInterval() const;
    int autoSaveShots() const;
    bool isInitialized() const;
    bool isAborted() const;
    bool isDummy() const;
    bool isLifWaiting() const;
    FtmwConfig ftmwConfig() const;
    PulseGenConfig pGenConfig() const;
    FlowConfig flowConfig() const;
    LifConfig lifConfig() const;
    bool isComplete() const;
    bool hardwareSuccess() const;
    QString errorString() const;
    QMap<QString,QList<QVariant>> timeDataMap() const;
    QString startLogMessage() const;
    QString endLogMessage() const;
    BlackChirp::LogMessageCode endLogMessageCode() const;
    QMap<QString, QPair<QVariant,QString>> headerMap() const;
    bool snapshotReady() const;

    void setTimeDataInterval(const int t);
    void setAutoSaveShotsInterval(const int s);
    void setInitialized();
    void setAborted();
    void setDummy();
    void setLifWaiting(bool wait);
    void setFtmwConfig(const FtmwConfig cfg);
    void setScopeConfig(const BlackChirp::FtmwScopeConfig &cfg);
    void setLifConfig(const LifConfig cfg);
    bool setFids(const QByteArray rawData);
    bool addFids(const QByteArray newData);
    bool addLifWaveform(const LifTrace t);
    void overrideTargetShots(const int target);
    void resetFids();
    void setPulseGenConfig(const PulseGenConfig c);
    void setFlowConfig(const FlowConfig c);
    void setErrorString(const QString str);
    void addTimeData(const QList<QPair<QString, QVariant> > dataList);
    void addTimeStamp();

    void setHardwareFailed();
    void incrementFtmw();

    void finalSave() const;
    bool saveHeader() const;
    bool saveChirpFile() const;
    void snapshot(int snapNum, const Experiment other) const;

private:
    QSharedDataPointer<ExperimentData> data;
};

class ExperimentData : public QSharedData
{
public:
    ExperimentData() : number(0), timeDataInterval(300), autoSaveShotsInterval(10000), isInitialized(false), isAborted(false),
        isDummy(false), hardwareSuccess(true), waitForLifSet(false), endLogMessageCode(BlackChirp::LogHighlight) {}

    int number;
    QDateTime startTime;
    int timeDataInterval;
    int autoSaveShotsInterval;
    bool isInitialized;
    bool isAborted;
    bool isDummy;
    bool hardwareSuccess;
    bool waitForLifSet;
    QString errorString;
    QString startLogMessage;
    QString endLogMessage;
    BlackChirp::LogMessageCode endLogMessageCode;

    FtmwConfig ftmwCfg;
    PulseGenConfig pGenCfg;
    FlowConfig flowCfg;
    LifConfig lifCfg;
    QMap<QString,QList<QVariant>> timeDataMap;
};

Q_DECLARE_METATYPE(Experiment)
Q_DECLARE_TYPEINFO(Experiment, Q_MOVABLE_TYPE);

#endif // EXPERIMENT_H
