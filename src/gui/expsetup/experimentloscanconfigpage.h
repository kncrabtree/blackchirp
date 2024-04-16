#ifndef EXPERIMENTLOSCANCONFIGPAGE_H
#define EXPERIMENTLOSCANCONFIGPAGE_H

#include "experimentconfigpage.h"

class QSpinBox;
class QDoubleSpinBox;
class QGroupBox;
class QCheckBox;

namespace BC::Key::WizLoScan {
static const QString key{"WizardLoScanPage"};
static const QString title{"LO Scan"};

static const QString shots{"shotsPerStep"};
static const QString sweeps{"targetSweeps"};

static const QString upStart{"upStart"};
static const QString upEnd{"upEnd"};
static const QString upNumMinor{"upNumMinor"};
static const QString upMinorStep{"upMinorStep"};
static const QString upNumMajor{"upNumMajor"};
static const QString upMajorStep{"upMajorStep"};

static const QString downStart{"downStart"};
static const QString downEnd{"downEnd"};
static const QString downNumMinor{"downNumMinor"};
static const QString downMinorStep{"downMinorStep"};
static const QString downNumMajor{"downNumMajor"};
static const QString downMajorStep{"downMajorStep"};

static const QString downFixed{"downFixed"};
static const QString constOffset{"downConstantOffset"};
}

class ExperimentLOScanConfigPage : public ExperimentConfigPage
{
    Q_OBJECT
public:
    ExperimentLOScanConfigPage(Experiment *exp, QWidget *parent = nullptr);

public slots:
    void startChanged(RfConfig::ClockType t, double val);
    void endChanged(RfConfig::ClockType t, double val);
    void minorStepChanged(RfConfig::ClockType t, int val);
    void minorStepSizeChanged(RfConfig::ClockType t, double val);
    void majorStepChanged(RfConfig::ClockType t, int val);
    void majorStepSizeChanged(RfConfig::ClockType t, double val);
    void fixedChanged(bool fixed);
    void constantOffsetChanged(bool co);

private:
    QSpinBox *p_upNumMinorBox, *p_downNumMinorBox, *p_upNumMajorBox, *p_downNumMajorBox, *p_shotsPerStepBox, *p_targetSweepsBox;
    QDoubleSpinBox *p_upStartBox, *p_downStartBox, *p_upEndBox, *p_downEndBox, *p_upMinorStepBox, *p_downMinorStepBox, *p_upMajorStepBox, *p_downMajorStepBox;
    QGroupBox *p_upBox, *p_downBox;
    QCheckBox *p_fixedDownLoBox, *p_constantDownOffsetBox;

    struct LoRanges {
        std::pair<double,double> upLoRange;
        std::pair<double,double> downLoRange;
    };

    double calculateMajorStepSize(RfConfig::ClockType t);
    LoRanges calculateLoRanges() const;

    // ExperimentConfigPage interface
public slots:
    void initialize() override;
    bool validate() override;
    void apply() override;
};

#endif // EXPERIMENTLOSCANCONFIGPAGE_H
