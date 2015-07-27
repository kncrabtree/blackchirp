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
#include "ioboardconfig.h"

class ExperimentData;

class Experiment
{
public:
    Experiment();
    Experiment(const Experiment &);
    Experiment &operator=(const Experiment &);
    Experiment(const int num, QString exptPath = QString(""));
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
    IOBoardConfig iobConfig() const;
    bool isComplete() const;
    bool hardwareSuccess() const;
    QString errorString() const;
    QMap<QString,QPair<QList<QVariant>,bool>> timeDataMap() const;
    QString startLogMessage() const;
    QString endLogMessage() const;
    BlackChirp::LogMessageCode endLogMessageCode() const;
    QMap<QString, QPair<QVariant,QString>> headerMap() const;
    bool snapshotReady();

    void setTimeDataInterval(const int t);
    void setAutoSaveShotsInterval(const int s);
    void setInitialized();
    void setAborted();
    void setDummy();
    void setLifWaiting(bool wait);
    void setFtmwConfig(const FtmwConfig cfg);
    void setScopeConfig(const BlackChirp::FtmwScopeConfig &cfg);
    void setLifConfig(const LifConfig cfg);
    void setIOBoardConfig(const IOBoardConfig cfg);
    bool setFidsData(const QList<QVector<qint64>> l);
    bool addFids(const QByteArray newData, int shift = 0);
    bool addLifWaveform(const LifTrace t);
    void overrideTargetShots(const int target);
    void resetFids();
    void setPulseGenConfig(const PulseGenConfig c);
    void setFlowConfig(const FlowConfig c);
    void setErrorString(const QString str);
    bool addTimeData(const QList<QPair<QString, QVariant> > dataList, bool plot);
    void addTimeStamp();
    void setValidationItems(const QMap<QString,BlackChirp::ValidationItem> m);

    void setHardwareFailed();
    void incrementFtmw();

    void finalSave() const;
    bool saveHeader() const;
    bool saveChirpFile() const;
    bool saveTimeFile() const;
    void snapshot(int snapNum, const Experiment other) const;

private:
    QSharedDataPointer<ExperimentData> data;
};

class ExperimentData : public QSharedData
{
public:
    ExperimentData() : number(0), timeDataInterval(300), autoSaveShotsInterval(10000), lastSnapshot(0), isInitialized(false),
        isAborted(false), isDummy(false), hardwareSuccess(true), waitForLifSet(false),
        endLogMessageCode(BlackChirp::LogHighlight) {}

    int number;
    QDateTime startTime;
    int timeDataInterval;
    int autoSaveShotsInterval;
    qint64 lastSnapshot;
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
    IOBoardConfig iobCfg;
    QMap<QString,QPair<QList<QVariant>,bool>> timeDataMap;
    QMap<QString,BlackChirp::ValidationItem> validationConditions;
};

Q_DECLARE_METATYPE(Experiment)

#endif // EXPERIMENT_H
