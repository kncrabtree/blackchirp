#include "wizardmotorscanconfigpage.h"

#include <QVBoxLayout>
#include <QMessageBox>

#include "motorscanconfigwidget.h"
#include "experimentwizard.h"

WizardMotorScanConfigPage::WizardMotorScanConfigPage(QWidget *parent) :
    QWizardPage(parent)
{
    setTitle(QString("Motor Scan Configuration"));
    setSubTitle(QString("Configure the parameters of the motor scan."));

    QVBoxLayout *vbl = new QVBoxLayout;

    p_mscw = new MotorScanConfigWidget(this);

    vbl->addWidget(p_mscw);

    setLayout(vbl);
}

MotorScan WizardMotorScanConfigPage::motorScan() const
{
    return p_mscw->toMotorScan();
}

bool WizardMotorScanConfigPage::validatePage()
{
    return p_mscw->validatePage();
}

int WizardMotorScanConfigPage::nextId() const
{
    return ExperimentWizard::PulseConfigPage;
}
