#include "experimentwizard.h"

#include "wizardstartpage.h"
#include "wizardchirpconfigpage.h"
#include "wizardftmwconfigpage.h"
#include "wizardsummarypage.h"
#include "wizardpulseconfigpage.h"
#include "wizardlifconfigpage.h"
#include "wizardvalidationpage.h"
#include "batchsingle.h"

ExperimentWizard::ExperimentWizard(QWidget *parent) :
    QWizard(parent)
{
    setWindowTitle(QString("Experiment Setup"));

    p_startPage = new WizardStartPage(this);
    p_chirpConfigPage = new WizardChirpConfigPage(this);
    p_ftmwConfigPage = new WizardFtmwConfigPage(this);
    p_pulseConfigPage = new WizardPulseConfigPage(this);
    p_validationPage = new WizardValidationPage(this);
    p_summaryPage = new WizardSummaryPage(this);
    p_lifConfigPage = new WizardLifConfigPage(this);
    connect(this,&ExperimentWizard::newTrace,p_lifConfigPage,&WizardLifConfigPage::newTrace);
    connect(this,&ExperimentWizard::scopeConfigChanged,p_lifConfigPage,&WizardLifConfigPage::scopeConfigChanged);
    connect(p_lifConfigPage,&WizardLifConfigPage::updateScope,this,&ExperimentWizard::updateScope);
    connect(p_lifConfigPage,&WizardLifConfigPage::lifColorChanged,this,&ExperimentWizard::lifColorChanged);


    setPage(StartPage,p_startPage);
    setPage(ChirpConfigPage,p_chirpConfigPage);
    setPage(FtmwConfigPage,p_ftmwConfigPage);
    setPage(PulseConfigPage,p_pulseConfigPage);
    setPage(LifConfigPage,p_lifConfigPage);
    setPage(ValidationPage,p_validationPage);
    setPage(SummaryPage,p_summaryPage);
}

ExperimentWizard::~ExperimentWizard()
{ 
}

void ExperimentWizard::setPulseConfig(const PulseGenConfig c)
{
    p_pulseConfigPage->setConfig(c);
}

void ExperimentWizard::setFlowConfig(const FlowConfig c)
{
    d_flowConfig = c;
}

Experiment ExperimentWizard::getExperiment() const
{
    Experiment exp;

    FtmwConfig ftc = p_ftmwConfigPage->getFtmwConfig();
    if(p_startPage->ftmwEnabled())
    {
        ftc.setEnabled();

        ChirpConfig cc = p_chirpConfigPage->getChirpConfig();
        ftc.setChirpConfig(cc);
    }
    LifConfig lc = p_lifConfigPage->getConfig();
    if(p_startPage->lifEnabled())
        lc.setEnabled();

    exp.setFtmwConfig(ftc);
    exp.setLifConfig(lc);
    exp.setPulseGenConfig(p_pulseConfigPage->getConfig());
    exp.setFlowConfig(d_flowConfig);
    exp.setIOBoardConfig(p_validationPage->getConfig());
    exp.setValidationItems(p_validationPage->getValidation());
    exp.setTimeDataInterval(p_startPage->auxDataInterval());
    exp.setAutoSaveShotsInterval(p_startPage->snapshotInterval());

    return exp;
}

BatchManager *ExperimentWizard::getBatchManager() const
{
    saveToSettings();
    Experiment e = getExperiment();
    if(e.lifConfig().isEnabled())
    {
        LifConfig lc = e.lifConfig();
        lc.allocateMemory();
        e.setLifConfig(lc);
    }

    return new BatchSingle(e);
}

void ExperimentWizard::saveToSettings() const
{
    p_startPage->saveToSettings();

    if(p_startPage->ftmwEnabled())
    {
        p_chirpConfigPage->saveToSettings();
        p_ftmwConfigPage->saveToSettings();
    }

    if(p_startPage->lifEnabled())
        p_lifConfigPage->saveToSettings();

    p_validationPage->saveToSettings();
}

