#ifndef EXPERIMENTLOSCANCONFIGPAGE_H
#define EXPERIMENTLOSCANCONFIGPAGE_H

#include "experimentconfigpage.h"

class QSpinBox;
class QDoubleSpinBox;
class QGroupBox;
class QCheckBox;

namespace BC::Key::WizLoScan {
inline constexpr QLatin1StringView key{"WizardLoScanPage"};
inline constexpr QLatin1StringView title{"LO Scan"};

inline constexpr QLatin1StringView shots{"shotsPerStep"};
inline constexpr QLatin1StringView sweeps{"targetSweeps"};

inline constexpr QLatin1StringView upStart{"upStart"};
inline constexpr QLatin1StringView upEnd{"upEnd"};
inline constexpr QLatin1StringView upNumMinor{"upNumMinor"};
inline constexpr QLatin1StringView upMinorStep{"upMinorStep"};
inline constexpr QLatin1StringView upNumMajor{"upNumMajor"};
inline constexpr QLatin1StringView upMajorStep{"upMajorStep"};

inline constexpr QLatin1StringView downStart{"downStart"};
inline constexpr QLatin1StringView downEnd{"downEnd"};
inline constexpr QLatin1StringView downNumMinor{"downNumMinor"};
inline constexpr QLatin1StringView downMinorStep{"downMinorStep"};
inline constexpr QLatin1StringView downNumMajor{"downNumMajor"};
inline constexpr QLatin1StringView downMajorStep{"downMajorStep"};

inline constexpr QLatin1StringView downFixed{"downFixed"};
inline constexpr QLatin1StringView constOffset{"downConstantOffset"};
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
