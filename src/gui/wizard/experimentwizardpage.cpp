#include <gui/wizard/experimentwizardpage.h>

ExperimentWizardPage::ExperimentWizardPage(const QString key, QWidget *parent) :
    QWizardPage(parent), SettingsStorage(key)
{
}

Experiment *ExperimentWizardPage::getExperiment() const
{
    return dynamic_cast<ExperimentWizard*>(wizard())->p_experiment;
}

int ExperimentWizardPage::startingFtmwPage() const
{
    auto e = getExperiment();
    switch(e->ftmwConfig()->d_type)
    {
    case FtmwConfig::LO_Scan:
        return ExperimentWizard::LoScanPage;
    case FtmwConfig::DR_Scan:
        return ExperimentWizard::DrScanPage;
    default:
        return ExperimentWizard::ChirpConfigPage;
    }
}

