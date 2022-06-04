#ifndef LIFCONFIG_H
#define LIFCONFIG_H

#include <QVector>
#include <QPointF>
#include <QMap>
#include <QVariant>

#include <data/storage/headerstorage.h>
#include <modules/lif/data/liftrace.h>
#include <modules/lif/hardware/lifdigitizer/lifdigitizerconfig.h>

namespace BC::Store::LIF {
static const QString key("LifConfig");
static const QString order("ScanOrder");
static const QString completeMode("CompleteMode");
static const QString dStart("DelayStart");
static const QString dStep("DelayStep");
static const QString dPoints("DelayPoints");
static const QString lStart("LaserStart");
static const QString lStep("LaserStep");
static const QString lPoints("LaserPoints");
static const QString shotsPerPoint("ShotsPerPoint");
}


class LifConfig : public HeaderStorage
{
    Q_GADGET
public:
    enum LifScanOrder {
        DelayFirst,
        LaserFirst
    };
    Q_ENUM(LifScanOrder)

    enum LifCompleteMode {
        StopWhenComplete,
        ContinueAveraging
    };
    Q_ENUM(LifCompleteMode)

    LifConfig();
    ~LifConfig() = default;

    bool d_enabled {false};
    bool d_complete {false};
    LifScanOrder d_order {DelayFirst};
    LifCompleteMode d_completeMode{ContinueAveraging};
    double d_delayStartUs {-1.0};
    double d_delayStepUs {0.0};
    int d_delayPoints {0};
    double d_laserPosStart {-1.0};
    double d_laserPosStep {0.0};
    int d_laserPosPoints {0};
    int d_lifGateStartPoint {-1};
    int d_lifGateEndPoint {-1};
    int d_refGateStartPoint {-1};
    int d_refGateEndPoint {-1};
    int d_currentDelayIndex {0};
    int d_currentFrequencyIndex {0};
    int d_shotsPerPoint {0};

    LifDigitizerConfig d_scopeConfig;
    QPoint d_lastUpdatedPoint;
    QVector<QVector<LifTrace>> d_lifData;

    bool isComplete() const;
    double currentDelay() const;
    double currentLaserPos() const;
    QPair<double,double> delayRange() const;
    QPair<double,double> laserRange() const;
    int totalShots() const;
    int completedShots() const;

    QPair<int,int> lifGate() const;
    QPair<int,int> refGate() const;
    QVector<QVector<LifTrace> > lifData() const;
    bool loadLifData(int num, QString path = QString(""));
    bool writeLifFile(int num) const;


    bool addWaveform(LifTrace t);



private:
    bool addTrace(const LifTrace t);
    void increment();

    // HeaderStorage interface
protected:
    void storeValues() override;
    void retrieveValues() override;

public:
    void prepareChildren() override;
};


#endif // LIFCONFIG_H
