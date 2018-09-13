#include "wizardrfconfigpage.h"

#include <QVBoxLayout>

#include "experimentwizard.h"

WizardRfConfigPage::WizardRfConfigPage(QWidget *parent) : QWizardPage(parent)
{
    setTitle(QString("Configure Clocks"));

    p_rfc = new RfConfigWidget(this);

    auto vbl = new QVBoxLayout(this);
    vbl->addWidget(p_rfc);

    setLayout(vbl);
}



void WizardRfConfigPage::initializePage()
{
    //TODO: get rfConfig from earlier page
    auto c = RfConfig::loadFromSettings();
    p_rfc->setRfConfig(c);
}

bool WizardRfConfigPage::validatePage()
{
    ///TODO: If segmented, check to make sure upconversion and downconversion LOs are set
    return true;
}

int WizardRfConfigPage::nextId() const
{
    return ExperimentWizard::ChirpConfigPage;
}
