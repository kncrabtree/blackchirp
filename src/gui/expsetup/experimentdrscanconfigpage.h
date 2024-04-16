#ifndef EXPERIMENTDRSCANCONFIGPAGE_H
#define EXPERIMENTDRSCANCONFIGPAGE_H

#include "experimentconfigpage.h"

class QDoubleSpinBox;
class QSpinBox;

namespace BC::Key::WizDR {
static const QString key{"WizardDrPage"};
static const QString title{"DR Scan"};
static const QString start{"startFreqMHz"};
static const QString step{"stepSizeMHz"};
static const QString numSteps{"numSteps"};
static const QString shots{"numShots"};
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
