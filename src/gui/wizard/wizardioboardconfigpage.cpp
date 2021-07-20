#include "wizardioboardconfigpage.h"

#include <QVBoxLayout>

WizardIOBoardConfigPage::WizardIOBoardConfigPage(QWidget *parent) : ExperimentWizardPage(BC::Key::WizIOB::key,parent)
{
    setTitle("IO Board Configuration");
    setSubTitle("Configure channels to be read as aux/validation data. Active analog channels will be plotted on the rolling data and aux data plots, and digital channels may be used as validation items on the next page.");

    auto vbl = new QVBoxLayout;

    p_iobWidget = new IOBoardConfigWidget;

    vbl->addWidget(p_iobWidget);

    setLayout(vbl);

}


void WizardIOBoardConfigPage::initializePage()
{
    auto e = getExperiment();
    if(e->d_number > 0)
    {
        if(e->iobConfig())
            p_iobWidget->setFromConfig(*e->iobConfig());
    }
}

bool WizardIOBoardConfigPage::validatePage()
{
    auto e = getExperiment();
    IOBoardConfig cfg;
    p_iobWidget->toConfig(cfg);
    e->setIOBoardConfig(cfg);

    return true;
}

int WizardIOBoardConfigPage::nextId() const
{
    return ExperimentWizard::ValidationPage;
}
