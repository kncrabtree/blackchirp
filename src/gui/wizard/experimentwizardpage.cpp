#include <src/gui/wizard/experimentwizardpage.h>

ExperimentWizardPage::ExperimentWizardPage(QWidget *parent) : QWizardPage(parent)
{
}

Experiment& ExperimentWizardPage::getExperiment() const
{
    return dynamic_cast<ExperimentWizard*>(wizard())->getExperiment();
}

int ExperimentWizardPage::startingFtmwPage() const
{
    auto e = getExperiment();
    switch(e.ftmwConfig().type())
    {
    case BlackChirp::FtmwLoScan:
        return ExperimentWizard::LoScanPage;
    default:
        return ExperimentWizard::ChirpConfigPage;
    }
}

