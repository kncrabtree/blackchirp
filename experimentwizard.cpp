#include "experimentwizard.h"

#include "wizardstartpage.h"
#include "wizardrfconfigpage.h"
#include "wizardchirpconfigpage.h"
#include "wizardftmwconfigpage.h"
#include "wizardsummarypage.h"
#include "wizardpulseconfigpage.h"
#include "wizardvalidationpage.h"
#include "batchsingle.h"

#ifdef BC_LIF
#include "wizardlifconfigpage.h"
#endif

#ifdef BC_MOTOR
#include "wizardmotorscanconfigpage.h"
#endif

ExperimentWizard::ExperimentWizard(QWidget *parent) :
    QWizard(parent)
{
    setWindowTitle(QString("Experiment Setup"));

    p_startPage = new WizardStartPage(this);
    p_rfConfigPage = new WizardRfConfigPage(this);
    p_chirpConfigPage = new WizardChirpConfigPage(this);
    p_ftmwConfigPage = new WizardFtmwConfigPage(this);
    p_pulseConfigPage = new WizardPulseConfigPage(this);
    p_validationPage = new WizardValidationPage(this);
    p_summaryPage = new WizardSummaryPage(this);

    setPage(StartPage,p_startPage);
    setPage(RfConfigPage,p_rfConfigPage);
    setPage(ChirpConfigPage,p_chirpConfigPage);
    setPage(FtmwConfigPage,p_ftmwConfigPage);
    setPage(PulseConfigPage,p_pulseConfigPage);
    setPage(ValidationPage,p_validationPage);
    setPage(SummaryPage,p_summaryPage);

#ifdef BC_LIF
    p_lifConfigPage = new WizardLifConfigPage(this);
    connect(this,&ExperimentWizard::newTrace,p_lifConfigPage,&WizardLifConfigPage::newTrace);
    connect(this,&ExperimentWizard::scopeConfigChanged,p_lifConfigPage,&WizardLifConfigPage::scopeConfigChanged);
    connect(p_lifConfigPage,&WizardLifConfigPage::updateScope,this,&ExperimentWizard::updateScope);
    connect(p_lifConfigPage,&WizardLifConfigPage::lifColorChanged,this,&ExperimentWizard::lifColorChanged);
    setPage(LifConfigPage,p_lifConfigPage);
#endif

#ifdef BC_MOTOR
    p_motorScanConfigPage = new WizardMotorScanConfigPage(this);
    setPage(MotorScanConfigPage,p_motorScanConfigPage);
#endif
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

        ///TODO: Get RF config from chirpconfigpage here!
//        ChirpConfig cc = p_chirpConfigPage->getChirpConfig();
//        ftc.setChirpConfig(cc);
    }

#ifdef BC_LIF
    LifConfig lc = p_lifConfigPage->getConfig();
    if(p_startPage->lifEnabled())
        lc.setEnabled();
    exp.setLifConfig(lc);
#endif

#ifdef BC_MOTOR
    MotorScan ms = p_motorScanConfigPage->motorScan();
    if(p_startPage->motorEnabled())
        ms.setEnabled();
    exp.setMotorScan(ms);
#endif

    exp.setFtmwConfig(ftc);
    exp.setPulseGenConfig(p_pulseConfigPage->getConfig());
    exp.setFlowConfig(d_flowConfig);
    exp.setIOBoardConfig(p_validationPage->getConfig());
    exp.setValidationItems(p_validationPage->getValidation());
    exp.setTimeDataInterval(p_startPage->auxDataInterval());
    exp.setAutoSaveShotsInterval(p_startPage->snapshotInterval());

    return exp;
}

bool ExperimentWizard::sleepWhenDone() const
{
    return field(QString("sleep")).toBool();
}
