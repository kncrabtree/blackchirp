#include "wizarddigitizerconfigpage.h"

#include <QVBoxLayout>
#include <QDialogButtonBox>

#include "digitizerconfigwidget.h"
#include "experimentwizard.h"

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
    emit experimentUpdate(e);
    return true;
}

int WizardDigitizerConfigPage::nextId() const
{
    return ExperimentWizard::PulseConfigPage;
}
