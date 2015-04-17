#include "experimentwizard.h"

ExperimentWizard::ExperimentWizard(QWidget *parent) :
    QWizard(parent)
{
    setWindowTitle(QString("Experiment Setup"));

    p_startPage = new WizardStartPage(this);
    p_chirpConfigPage = new WizardChirpConfigPage(this);
    p_ftmwConfigPage = new WizardFtmwConfigPage(this);
    p_pulseConfigPage = new WizardPulseConfigPage(this);
    p_summaryPage = new WizardSummaryPage(this);


    setPage(StartPage,p_startPage);
    setPage(ChirpConfigPage,p_chirpConfigPage);
    setPage(FtmwConfigPage,p_ftmwConfigPage);
    setPage(PulseConfigPage,p_pulseConfigPage);
    setPage(SummaryPage,p_summaryPage);
}

ExperimentWizard::~ExperimentWizard()
{ 
}

void ExperimentWizard::setPulseConfig(const PulseGenConfig c)
{
    p_pulseConfigPage->setConfig(c);
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
    exp.setPulseGenConfig(p_pulseConfigPage->getConfig());

    return exp;
}

void ExperimentWizard::saveToSettings()
{
    if(p_startPage->ftmwEnabled())
    {
        p_chirpConfigPage->saveToSettings();
        p_ftmwConfigPage->saveToSettings();
    }
}
