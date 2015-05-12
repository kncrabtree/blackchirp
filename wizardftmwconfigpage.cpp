#include "wizardftmwconfigpage.h"

#include <QVBoxLayout>
#include <QDialogButtonBox>

#include "ftmwconfigwidget.h"
#include "experimentwizard.h"

WizardFtmwConfigPage::WizardFtmwConfigPage(QWidget *parent) :
    QWizardPage(parent)
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
    p_ftc->lockFastFrame(field(QString("numChirps")).toInt());
}

bool WizardFtmwConfigPage::validatePage()
{
    return true;
}

int WizardFtmwConfigPage::nextId() const
{
    return ExperimentWizard::PulseConfigPage;
}

FtmwConfig WizardFtmwConfigPage::getFtmwConfig() const
{
    return p_ftc->getConfig();
}

void WizardFtmwConfigPage::saveToSettings()
{
    p_ftc->saveToSettings();
}
