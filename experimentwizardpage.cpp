#include "experimentwizardpage.h"

ExperimentWizardPage::ExperimentWizardPage(QWidget *parent) : QWizardPage(parent)
{
}

Experiment ExperimentWizardPage::getExperiment() const
{
    return dynamic_cast<ExperimentWizard*>(wizard())->getExperiment();
}

