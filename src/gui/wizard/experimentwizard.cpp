#include <src/gui/wizard/experimentwizard.h>

#include <src/gui/wizard/experimentwizardpage.h>
#include <src/gui/wizard/wizardstartpage.h>
#include <src/gui/wizard/wizardloscanconfigpage.h>
#include <src/gui/wizard/wizardrfconfigpage.h>
#include <src/gui/wizard/wizardchirpconfigpage.h>
#include <src/gui/wizard/wizarddigitizerconfigpage.h>
#include <src/gui/wizard/wizardsummarypage.h>
#include <src/gui/wizard/wizardpulseconfigpage.h>
#include <src/gui/wizard/wizardvalidationpage.h>
#include <src/acquisition/batch/batchsingle.h>

#ifdef BC_LIF
#include <src/gui/wizard/wizardlifconfigpage.h>
#endif

#ifdef BC_MOTOR
#include <src/gui/wizard/wizardmotorscanconfigpage.h>
#endif

ExperimentWizard::ExperimentWizard(QWidget *parent) :
    QWizard(parent)
{
    setWindowTitle(QString("Experiment Setup"));

    auto startPage = new WizardStartPage(this);
    d_pages << startPage;

    auto loScanConfigPage = new WizardLoScanConfigPage(this);
    d_pages << loScanConfigPage;

    auto rfConfigPage = new WizardRfConfigPage(this);
    d_pages << rfConfigPage;

    auto chirpConfigPage = new WizardChirpConfigPage(this);
    d_pages << chirpConfigPage;

    auto digitizerConfigPage = new WizardDigitizerConfigPage(this);
    d_pages << digitizerConfigPage;

    auto pulseConfigPage = new WizardPulseConfigPage(this);
    d_pages << pulseConfigPage;

    auto validationPage = new WizardValidationPage(this);
    d_pages << validationPage;

    auto summaryPage = new WizardSummaryPage(this);
    d_pages << summaryPage;

    setPage(StartPage,startPage);
    setPage(LoScanPage,loScanConfigPage);
    setPage(RfConfigPage,rfConfigPage);
    setPage(ChirpConfigPage,chirpConfigPage);
    setPage(DigitizerConfigPage,digitizerConfigPage);
    setPage(PulseConfigPage,pulseConfigPage);
    setPage(ValidationPage,validationPage);
    setPage(SummaryPage,summaryPage);


#ifdef BC_LIF
    auto lifConfigPage = new WizardLifConfigPage(this);
    p_lifConfigPage = lifConfigPage;
    d_pages << lifConfigPage;
    connect(this,&ExperimentWizard::newTrace,lifConfigPage,&WizardLifConfigPage::newTrace);
    connect(this,&ExperimentWizard::scopeConfigChanged,lifConfigPage,&WizardLifConfigPage::scopeConfigChanged);
    connect(lifConfigPage,&WizardLifConfigPage::updateScope,this,&ExperimentWizard::updateScope);
    connect(lifConfigPage,&WizardLifConfigPage::lifColorChanged,this,&ExperimentWizard::lifColorChanged);
    connect(lifConfigPage,&WizardLifConfigPage::laserPosUpdate,this,&ExperimentWizard::laserPosUpdate);
    setPage(LifConfigPage,lifConfigPage);
#endif

#ifdef BC_MOTOR
    auto motorScanConfigPage = new WizardMotorScanConfigPage(this);
    d_pages << motorScanConfigPage;
    setPage(MotorScanConfigPage,motorScanConfigPage);
#endif

    for(int i=0; i<d_pages.size(); i++)
        connect(d_pages.at(i),&ExperimentWizardPage::experimentUpdate,this,&ExperimentWizard::updateExperiment);

    d_experiment = Experiment::loadFromSettings();
}

ExperimentWizard::~ExperimentWizard()
{ 
}

void ExperimentWizard::setPulseConfig(const PulseGenConfig c)
{
    d_experiment.setPulseGenConfig(c);
}

void ExperimentWizard::setFlowConfig(const FlowConfig c)
{
    d_experiment.setFlowConfig(c);
}

Experiment ExperimentWizard::getExperiment() const
{
    return d_experiment;
}

QSize ExperimentWizard::sizeHint() const
{
    return {1000,700};
}

#ifdef BC_LIF
void ExperimentWizard::setCurrentLaserPos(double pos)
{
    dynamic_cast<WizardLifConfigPage*>(p_lifConfigPage)->setLaserPos(pos);
}
#endif


