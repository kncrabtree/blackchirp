#ifndef EXPERIMENT_H
#define EXPERIMENT_H

#include <QSharedDataPointer>
#include <QPair>
#include <QDateTime>
#include <QMetaType>

#include <src/data/experiment/ftmwconfig.h>
#include <src/data/datastructs.h>
#include <src/data/experiment/pulsegenconfig.h>
#include <src/data/experiment/flowconfig.h>
#include <src/data/experiment/ioboardconfig.h>

//these are included because they define datatypes needed by qRegisterMetaType in main.cpp
#include <src/data/analysis/ft.h>
#include <src/data/analysis/ftworker.h>

#ifdef BC_LIF
#include <src/modules/lif/data/lifconfig.h>
#endif

#ifdef BC_MOTOR
#include <src/modules/motor/data/motorscan.h>
#endif


class Experiment
{
public:
    Experiment();
    Experiment(const Experiment &) = default;
    Experiment& operator=(const Experiment &) = default;
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
    void finalizeFtmwSnapshots(const FtmwConfig final);

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
    bool saveClockFile() const;
    bool saveTimeFile() const;
    QString timeDataText() const;
    void snapshot(int snapNum, const Experiment other) const;
    void exportAscii(const QString fileName) const;

    void saveToSettings() const;
    static Experiment loadFromSettings();

private:
    int d_number;
    QDateTime d_startTime;
    int d_timeDataInterval;
    int d_autoSaveShotsInterval;
    qint64 d_lastSnapshot;
    bool d_isInitialized;
    bool d_isAborted;
    bool d_isDummy;
    bool d_hardwareSuccess;
    QString d_errorString;
    QString d_startLogMessage;
    QString d_endLogMessage;
    BlackChirp::LogMessageCode d_endLogMessageCode;

    FtmwConfig d_ftmwCfg;
    PulseGenConfig d_pGenCfg;
    FlowConfig d_flowCfg;
    IOBoardConfig d_iobCfg;
    QMap<QString,QPair<QList<QVariant>,bool>> d_timeDataMap;
    QMap<QString,BlackChirp::ValidationItem> d_validationConditions;

    QString d_path;

#ifdef BC_LIF
    LifConfig d_lifCfg;
    bool d_waitForLifSet;
#endif

#ifdef BC_MOTOR
    MotorScan d_motorScan;
#endif
};

Q_DECLARE_METATYPE(Experiment)

#endif // EXPERIMENT_H
