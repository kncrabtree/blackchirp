#include "wizardftmwconfigpage.h"

#include <QVBoxLayout>
#include <QDialogButtonBox>

#include "ftmwconfigwidget.h"
#include "experimentwizard.h"

WizardFtmwConfigPage::WizardFtmwConfigPage(QWidget *parent) :
    ExperimentWizardPage(parent)
{
    setTitle(QString("Configure FTMW Acquisition"));

    QVBoxLayout *vbl = new QVBoxLayout(this);
    p_ftc = new FtmwConfigWidget(this);

    vbl->addWidget(p_ftc);

    setLayout(vbl);
}

WizardFtmwConfigPage::~WizardFtmwConfigPage()
{
}



void WizardFtmwConfigPage::initializePage()
{
    ///TODO: Be more flexible here
    auto e = getExperiment();
    p_ftc->setFromConfig(e.ftmwConfig());
}

bool WizardFtmwConfigPage::validatePage()
{
    auto e = getExperiment();
    e.setFtmwConfig(p_ftc->getConfig());
    emit experimentUpdate(e);
    return true;
}

int WizardFtmwConfigPage::nextId() const
{
    return ExperimentWizard::PulseConfigPage;
}
