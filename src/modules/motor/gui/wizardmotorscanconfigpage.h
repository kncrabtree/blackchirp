#ifndef WIZARDMOTORSCANCONFIGPAGE_H
#define WIZARDMOTORSCANCONFIGPAGE_H

#include <gui/wizard/experimentwizardpage.h>

class MotorScanConfigWidget;

namespace BC::Key::WizMotor {
static const QString key("WizardMotorPage");
}

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
