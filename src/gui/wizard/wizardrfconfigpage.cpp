#include "wizardrfconfigpage.h"

#include <QVBoxLayout>

WizardRfConfigPage::WizardRfConfigPage(QWidget *parent) : ExperimentWizardPage(parent)
{
    setTitle(QString("Configure Clocks"));
    setSubTitle(QString("Configure the clock setup. If the experiment involves multiple clock frequencies (e.g., an LO scan), the frequencies will be set automatically."));

    p_rfc = new RfConfigWidget(this);

    auto vbl = new QVBoxLayout(this);
    vbl->addWidget(p_rfc);

    setLayout(vbl);
}



void WizardRfConfigPage::initializePage()
{
    auto e = getExperiment();
    p_rfc->setRfConfig(e.ftmwConfig().rfConfig());
}

bool WizardRfConfigPage::validatePage()
{
    ///TODO: If segmented, check to make sure upconversion and downconversion LOs are set

    auto e = getExperiment();
    e.setRfConfig(p_rfc->getRfConfig());
    
    return true;
}

int WizardRfConfigPage::nextId() const
{
    return startingFtmwPage();
}
