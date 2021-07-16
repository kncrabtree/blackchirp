#include <gui/wizard/experimentwizard.h>

#include <gui/wizard/experimentwizardpage.h>
#include <gui/wizard/wizardstartpage.h>
#include <gui/wizard/wizardloscanconfigpage.h>
#include <gui/wizard/wizarddrscanconfigpage.h>
#include <gui/wizard/wizardrfconfigpage.h>
#include <gui/wizard/wizardchirpconfigpage.h>
#include <gui/wizard/wizarddigitizerconfigpage.h>
#include <gui/wizard/wizardsummarypage.h>
#include <gui/wizard/wizardpulseconfigpage.h>
#include <gui/wizard/wizardioboardconfigpage.h>
#include <gui/wizard/wizardvalidationpage.h>
#include <acquisition/batch/batchsingle.h>


#ifdef BC_LIF
#include <modules/lif/gui/wizardlifconfigpage.h>
#endif

#ifdef BC_MOTOR
#include <modules/motor/gui/wizardmotorscanconfigpage.h>
#endif

ExperimentWizard::ExperimentWizard(int num, QWidget *parent) :
    QWizard(parent)
{
#pragma message("How to initialize from previous experiment?")
    (void)num;
    experiment = std::make_shared<Experiment>();

    setWindowTitle(QString("Experiment Setup"));

    auto startPage = new WizardStartPage(this);
    d_pages << startPage;

    auto loScanConfigPage = new WizardLoScanConfigPage(this);
    d_pages << loScanConfigPage;

    auto drScanConfigPage = new WizardDrScanConfigPage(this);
    d_pages << drScanConfigPage;

    auto rfConfigPage = new WizardRfConfigPage(this);
    d_pages << rfConfigPage;

    auto chirpConfigPage = new WizardChirpConfigPage(this);
    d_pages << chirpConfigPage;

    auto digitizerConfigPage = new WizardDigitizerConfigPage(this);
    d_pages << digitizerConfigPage;

    auto pulseConfigPage = new WizardPulseConfigPage(this);
    d_pages << pulseConfigPage;

    auto iobConfigPage = new WizardIOBoardConfigPage(this);
    d_pages << iobConfigPage;

    auto validationPage = new WizardValidationPage(this);
    d_pages << validationPage;

    auto summaryPage = new WizardSummaryPage(this);
    d_pages << summaryPage;

    setPage(StartPage,startPage);
    setPage(LoScanPage,loScanConfigPage);
    setPage(DrScanPage,drScanConfigPage);
    setPage(RfConfigPage,rfConfigPage);
    setPage(ChirpConfigPage,chirpConfigPage);
    setPage(DigitizerConfigPage,digitizerConfigPage);
    setPage(PulseConfigPage,pulseConfigPage);
    setPage(IOBoardConfigPage,iobConfigPage);
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
}

ExperimentWizard::~ExperimentWizard()
{ 
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




void ExperimentWizard::reject()
{
    for(auto page : d_pages)
        page->discardChanges();

    QDialog::reject();
}
