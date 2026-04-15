#ifndef EXPERIMENTDRSCANCONFIGPAGE_H
#define EXPERIMENTDRSCANCONFIGPAGE_H

#include "experimentconfigpage.h"

class QDoubleSpinBox;
class QSpinBox;

namespace BC::Key::WizDR {
inline constexpr QLatin1StringView key{"WizardDrPage"};
inline constexpr QLatin1StringView title{"DR Scan"};
inline constexpr QLatin1StringView start{"startFreqMHz"};
inline constexpr QLatin1StringView step{"stepSizeMHz"};
inline constexpr QLatin1StringView numSteps{"numSteps"};
inline constexpr QLatin1StringView shots{"numShots"};
}


class ExperimentDRScanConfigPage : public ExperimentConfigPage
{
    Q_OBJECT
public:
    ExperimentDRScanConfigPage(Experiment *exp, QWidget *parent = nullptr);

    // ExperimentConfigPage interface
public slots:
    void initialize() override;
    bool validate() override;
    void apply() override;

public slots:
    void updateEndBox();

private:
    QDoubleSpinBox *p_startBox, *p_stepSizeBox, *p_endBox;
    QSpinBox *p_numStepsBox, *p_shotsBox;
};

#endif // EXPERIMENTDRSCANCONFIGPAGE_H
