#include "experimentwizard.h"

ExperimentWizard::ExperimentWizard(QWidget *parent) :
    QWizard(parent)
{
    setWindowTitle(QString("Experiment Setup"));

    p_startPage = new WizardStartPage(this);
    p_chirpConfigPage = new WizardChirpConfigPage(this);
    p_ftmwConfigPage = new WizardFtmwConfigPage(this);
    p_summaryPage = new WizardSummaryPage(this);

    setPage(StartPage,p_startPage);
    setPage(ChirpConfigPage,p_chirpConfigPage);
    setPage(FtmwConfigPage,p_ftmwConfigPage);
    setPage(SummaryPage,p_summaryPage);
}

ExperimentWizard::~ExperimentWizard()
{ 
}

Experiment ExperimentWizard::getExperiment() const
{
    Experiment exp;

    FtmwConfig ftc = p_ftmwConfigPage->getFtmwConfig();
    if(p_startPage->ftmwEnabled())
        ftc.setEnabled();

    ChirpConfig cc = p_chirpConfigPage->getChirpConfig();
    ftc.setChirpConfig(cc);

    exp.setFtmwConfig(ftc);
    return exp;
}

