#include "wizardmotorscanconfigpage.h"

#include <QVBoxLayout>
#include <QMessageBox>

#include <src/modules/motor/gui/motorscanconfigwidget.h>
#include <src/gui/wizard/experimentwizard.h>

WizardMotorScanConfigPage::WizardMotorScanConfigPage(QWidget *parent) :
    ExperimentWizardPage(parent)
{
    setTitle(QString("Motor Scan Configuration"));
    setSubTitle(QString("Configure the parameters of the motor scan."));

    QVBoxLayout *vbl = new QVBoxLayout;

    p_mscw = new MotorScanConfigWidget(this);

    vbl->addWidget(p_mscw);

    setLayout(vbl);
}

void WizardMotorScanConfigPage::initializePage()
{
    auto e = getExperiment();
    p_mscw->setFromMotorScan(e.motorScan());
}

bool WizardMotorScanConfigPage::validatePage()
{
    if(p_mscw->validatePage())
    {
        auto e = getExperiment();
        e.setMotorScan(p_mscw->toMotorScan());
        emit experimentUpdate(e);
        return true;
    }
    return false;
}

int WizardMotorScanConfigPage::nextId() const
{
    return ExperimentWizard::PulseConfigPage;
}
