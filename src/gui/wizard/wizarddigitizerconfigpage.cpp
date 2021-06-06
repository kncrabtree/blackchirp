#include "wizarddigitizerconfigpage.h"

#include <QVBoxLayout>
#include <QDialogButtonBox>

#include <src/gui/widget/digitizerconfigwidget.h>
#include <src/gui/wizard/experimentwizard.h>

WizardDigitizerConfigPage::WizardDigitizerConfigPage(QWidget *parent) :
    ExperimentWizardPage(parent)
{
    setTitle(QString("Configure Digitizer"));

    QVBoxLayout *vbl = new QVBoxLayout(this);
    p_dc = new DigitizerConfigWidget(this);

    vbl->addWidget(p_dc);

    setLayout(vbl);
}

WizardDigitizerConfigPage::~WizardDigitizerConfigPage()
{
}



void WizardDigitizerConfigPage::initializePage()
{
    ///TODO: Be more flexible here
    auto e = getExperiment();
    p_dc->setFromConfig(e.ftmwConfig());
}

bool WizardDigitizerConfigPage::validatePage()
{
    auto e = getExperiment();
    e.setFtmwConfig(p_dc->getConfig());
    
    return true;
}

int WizardDigitizerConfigPage::nextId() const
{
    return ExperimentWizard::PulseConfigPage;
}