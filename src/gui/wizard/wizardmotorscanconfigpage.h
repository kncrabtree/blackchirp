#ifndef WIZARDMOTORSCANCONFIGPAGE_H
#define WIZARDMOTORSCANCONFIGPAGE_H

#include <src/gui/wizard/experimentwizardpage.h>

class MotorScanConfigWidget;

class WizardMotorScanConfigPage : public ExperimentWizardPage
{
public:
    WizardMotorScanConfigPage(QWidget *parent = nullptr);

    void initializePage();
    bool validatePage();
    int nextId() const;

private:
    MotorScanConfigWidget *p_mscw;

};

#endif // WIZARDMOTORSCANCONFIGPAGE_H
