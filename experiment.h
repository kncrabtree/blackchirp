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
#include "ioboardconfig.h"

//these are included because they define datatypes needed by qRegisterMetaType in main.cpp
#include "ft.h"
#include "ftworker.h"

#ifdef BC_LIF
#include "lifconfig.h"
#endif

#ifdef BC_MOTOR
#include "motorscan.h"
#endif


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
    FtmwConfig ftmwConfig() const;
    PulseGenConfig pGenConfig() const;
    FlowConfig flowConfig() const;
    IOBoardConfig iobConfig() const;
    bool isComplete() const;
    bool hardwareSuccess() const;
    QString errorString() const;
    QMap<QString,QPair<QList<QVariant>,bool>> timeDataMap() const;
    QString startLogMessage() const;
    QString endLogMessage() const;
    BlackChirp::LogMessageCode endLogMessageCode() const;
    QMap<QString, QPair<QVariant,QString>> headerMap() const;
    QMap<QString,BlackChirp::ValidationItem> validationItems() const;
    bool snapshotReady();

    void setTimeDataInterval(const int t);
    void setAutoSaveShotsInterval(const int s);
    void setInitialized();
    void setAborted();
    void setDummy();
    void setFtmwConfig(const FtmwConfig cfg);
    void setFtmwEnabled(bool en = true);
    void setScopeConfig(const BlackChirp::FtmwScopeConfig &cfg);
    void setRfConfig(const RfConfig cfg);
    void setIOBoardConfig(const IOBoardConfig cfg);
    bool setFidsData(const QList<QVector<qint64>> l);
    bool addFids(const QByteArray newData, int shift = 0);
    void overrideTargetShots(const int target);
    void resetFids();
    void setPulseGenConfig(const PulseGenConfig c);
    void setFlowConfig(const FlowConfig c);
    void setErrorString(const QString str);
    bool addTimeData(const QList<QPair<QString, QVariant> > dataList, bool plot);
    void addTimeStamp();
    void setValidationItems(const QMap<QString,BlackChirp::ValidationItem> m);
    void addValidationItem(const QString key, const double min, const double max);
    void addValidationItem(const BlackChirp::ValidationItem &i);

#ifdef BC_LIF
    bool isLifWaiting() const;
    LifConfig lifConfig() const;
    void setLifEnabled(bool en = true);
    void setLifWaiting(bool wait);
    void setLifConfig(const LifConfig cfg);
    bool addLifWaveform(const LifTrace t);
#endif

#ifdef BC_MOTOR
    MotorScan motorScan() const;
    void setMotorEnabled(bool en = true);
    void setMotorScan(const MotorScan s);
    bool addMotorTrace(const QVector<double> d);
#endif

    void setHardwareFailed();
    /**
     * @brief incrementFtmw
     * @return Boolean that indicates a new segment needs to start
     */
    bool incrementFtmw();
    void setFtmwClocksReady();

    void finalSave() const;
    bool saveHeader() const;
    bool saveChirpFile() const;
    bool saveTimeFile() const;
    QString timeDataText() const;
    void snapshot(int snapNum, const Experiment other) const;
    void exportAscii(const QString fileName) const;

    void saveToSettings() const;
    static Experiment loadFromSettings();

private:
    QSharedDataPointer<ExperimentData> data;
};

class ExperimentData : public QSharedData
{
public:
    ExperimentData() : number(0), timeDataInterval(300), autoSaveShotsInterval(10000), lastSnapshot(0), isInitialized(false),
        isAborted(false), isDummy(false), hardwareSuccess(true), endLogMessageCode(BlackChirp::LogHighlight)
#ifdef BC_LIF
    ,  waitForLifSet(false)
#endif
    {}

    int number;
    QDateTime startTime;
    int timeDataInterval;
    int autoSaveShotsInterval;
    qint64 lastSnapshot;
    bool isInitialized;
    bool isAborted;
    bool isDummy;
    bool hardwareSuccess;
    QString errorString;
    QString startLogMessage;
    QString endLogMessage;
    BlackChirp::LogMessageCode endLogMessageCode;

    FtmwConfig ftmwCfg;
    PulseGenConfig pGenCfg;
    FlowConfig flowCfg;
    IOBoardConfig iobCfg;
    QMap<QString,QPair<QList<QVariant>,bool>> timeDataMap;
    QMap<QString,BlackChirp::ValidationItem> validationConditions;

#ifdef BC_LIF
    LifConfig lifCfg;
    bool waitForLifSet;
#endif

#ifdef BC_MOTOR
    MotorScan motorScan;
#endif
};

Q_DECLARE_METATYPE(Experiment)

#endif // EXPERIMENT_H
