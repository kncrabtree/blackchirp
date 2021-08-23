#ifndef WIZARDDRSCANCONFIGPAGE_H
#define WIZARDDRSCANCONFIGPAGE_H

#include "experimentwizardpage.h"

class QDoubleSpinBox;
class QSpinBox;

namespace BC::Key::WizDR {
static const QString key("WizardDrPage");
static const QString start("startFreqMHz");
static const QString step("stepSizeMHz");
static const QString numSteps("numSteps");
static const QString shots("numShots");
}

class WizardDrScanConfigPage : public ExperimentWizardPage
{
    Q_OBJECT
public:
    WizardDrScanConfigPage(QWidget *parent=nullptr);

    // QWizardPage interface
public:
    void initializePage() override;
    bool validatePage() override;
    bool isComplete() const override;
    int nextId() const override;

public slots:
    void updateEndBox();

private:
    QDoubleSpinBox *p_startBox, *p_stepSizeBox, *p_endBox;
    QSpinBox *p_numStepsBox, *p_shotsBox;
};

#endif // WIZARDDRSCANCONFIGPAGE_H
