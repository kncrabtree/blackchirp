#ifndef LIFCONFIG_H
#define LIFCONFIG_H

#include <QVector>
#include <QPointF>
#include <QMap>
#include <QVariant>
#include <memory>

#include <data/storage/headerstorage.h>
#include <data/experiment/experimentobjective.h>
#include <modules/lif/data/lifstorage.h>
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

namespace BC::Config::Exp {
static const QString lifType{"LifType"};
}

class LifConfig : public ExperimentObjective, public HeaderStorage
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

    bool d_complete {false};
    LifScanOrder d_order {DelayFirst};
    LifCompleteMode d_completeMode{ContinueAveraging};
    bool d_disableFlashlamp{true};

    double d_delayStartUs {-1.0};
    double d_delayStepUs {0.0};
    int d_delayPoints {0};

    double d_laserPosStart {-1.0};
    double d_laserPosStep {0.0};
    int d_laserPosPoints {0};

    LifTrace::LifProcSettings d_procSettings;

    int d_shotsPerPoint {0};

    LifDigitizerConfig &scopeConfig() { return *ps_scopeConfig; }
    const LifDigitizerConfig &scopeConfig() const { return std::as_const(*ps_scopeConfig); }

    bool isComplete() const override;
    double currentDelay() const;
    double currentLaserPos() const;
    QPair<double,double> delayRange() const;
    QPair<double,double> laserRange() const;
    int targetShots() const;
    int completedShots() const;
    QPair<int,int> lifGate() const;
    QPair<int,int> refGate() const;
    std::shared_ptr<LifStorage> storage() { return ps_storage; }

    void addWaveform(const QVector<qint8> d);
    void loadLifData();



private:
    std::shared_ptr<LifStorage> ps_storage;
    std::shared_ptr<LifDigitizerConfig> ps_scopeConfig;
    int d_currentDelayIndex {0};
    int d_currentLaserIndex {0};
    int d_completedSweeps{0};

    // HeaderStorage interface
protected:
    void storeValues() override;
    void retrieveValues() override;

public:
    void prepareChildren() override;

    // ExperimentObjective interface
public:
    bool initialize() override;
    bool advance() override;
    void hwReady() override;
    int perMilComplete() const override;
    bool indefinite() const override;
    bool abort() override;
    QString objectiveKey() const override;
    void cleanupAndSave() override;
};


#endif // LIFCONFIG_H
