#include "wizardrfconfigpage.h"

#include <QVBoxLayout>

WizardRfConfigPage::WizardRfConfigPage(QWidget *parent) : ExperimentWizardPage(parent)
{
    setTitle(QString("Configure Clocks"));

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
    emit experimentUpdate(e);
    return true;
}

int WizardRfConfigPage::nextId() const
{
    return ExperimentWizard::ChirpConfigPage;
}
