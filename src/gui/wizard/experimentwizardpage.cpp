#include <gui/wizard/experimentwizardpage.h>

ExperimentWizardPage::ExperimentWizardPage(const QString key, QWidget *parent) :
    QWizardPage(parent), SettingsStorage(key)
{
}

std::shared_ptr<Experiment> ExperimentWizardPage::getExperiment() const
{
    return dynamic_cast<ExperimentWizard*>(wizard())->experiment;
}

int ExperimentWizardPage::startingFtmwPage() const
{
    auto e = getExperiment();
    switch(e->ftmwConfig().type())
    {
    case BlackChirp::FtmwLoScan:
        return ExperimentWizard::LoScanPage;
    case BlackChirp::FtmwDrScan:
        return ExperimentWizard::DrScanPage;
    default:
        return ExperimentWizard::ChirpConfigPage;
    }
}

