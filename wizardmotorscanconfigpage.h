#ifndef WIZARDMOTORSCANCONFIGPAGE_H
#define WIZARDMOTORSCANCONFIGPAGE_H

#include <QWizardPage>

#include "motorscan.h"

class MotorScanConfigWidget;

class WizardMotorScanConfigPage : public QWizardPage
{
public:
    WizardMotorScanConfigPage(QWidget *parent = nullptr);

    MotorScan motorScan() const;

    bool validatePage();
    int nextId() const;

private:
    MotorScanConfigWidget *p_mscw;

};

#endif // WIZARDMOTORSCANCONFIGPAGE_H
