#ifndef LIFCONFIG_H
#define LIFCONFIG_H

#include <QSharedDataPointer>

#include <QDataStream>
#include <QList>
#include <QVector>
#include <QPointF>
#include <QMap>
#include <QVariant>

#include "datastructs.h"
#include "liftrace.h"


class LifConfigData;

class LifConfig
{
public:
    LifConfig();
    LifConfig(const LifConfig &);
    LifConfig &operator=(const LifConfig &);
    ~LifConfig();   

    bool isEnabled() const;
    bool isComplete() const;
    double currentDelay() const;
    double currentLaserPos() const;
    QPair<double,double> delayRange() const;
    double delayStep() const;
    QPair<double,double> laserRange() const;
    double laserStep() const;
    int numDelayPoints() const;
    int numLaserPoints() const;
    int shotsPerPoint() const;
    int totalShots() const;
    int completedShots() const;
    BlackChirp::LifScopeConfig scopeConfig() const;
    BlackChirp::LifScanOrder order() const;
    BlackChirp::LifCompleteMode completeMode() const;
    QPair<int,int> lifGate() const;
    QPair<int,int> refGate() const;
    QList<QList<LifTrace> > lifData() const;
    QMap<QString,QPair<QVariant,QString> > headerMap() const;
    void parseLine(QString key, QVariant val);
    bool loadLifData(int num, QString path = QString(""));
    bool writeLifFile(int num) const;

    void setEnabled(bool en = true);
    void setLifGate(int start, int end);
    void setLifGate(const QPair<int, int> p);
    void setRefGate(int start, int end);
    void setRefGate(const QPair<int,int> p);
    void setDelayParameters(double start, double step, int count);
    void setLaserParameters(double start,  double step, int count);
    void setOrder(BlackChirp::LifScanOrder o);
    void setCompleteMode(BlackChirp::LifCompleteMode mode);
    void setScopeConfig(BlackChirp::LifScopeConfig c);
    void setShotsPerPoint(int pts);
    bool addWaveform(LifTrace t);

    void saveToSettings() const;
    static LifConfig loadFromSettings();


private:
    QSharedDataPointer<LifConfigData> data;

    bool addTrace(const LifTrace t);
    void increment();
};


class LifConfigData : public QSharedData
{
public:
    LifConfigData() = default;

    bool enabled {false};
    bool complete {false};
    BlackChirp::LifScanOrder order {BlackChirp::LifOrderDelayFirst};
    BlackChirp::LifCompleteMode completeMode{BlackChirp::LifContinueUntilExperimentComplete};
    double delayStartUs {-1.0};
    double delayStepUs {0.0};
    int delayPoints {0};
    double laserPosStart {-1.0};
    double laserPosStep {0.0};
    int laserPosPoints {0};
    int lifGateStartPoint {-1};
    int lifGateEndPoint {-1};
    int refGateStartPoint {-1};
    int refGateEndPoint {-1};
    int currentDelayIndex {0};
    int currentFrequencyIndex {0};
    int shotsPerPoint {0};
    QPoint lastUpdatedPoint;

    BlackChirp::LifScopeConfig scopeConfig;
    QList<QList<LifTrace>> lifData;

};

Q_DECLARE_METATYPE(LifConfig)


#endif // LIFCONFIG_H
