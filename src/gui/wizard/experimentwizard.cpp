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

#include <hardware/optional/pulsegenerator/pulsegenerator.h>
#include <hardware/optional/ioboard/ioboard.h>


#ifdef BC_LIF
#include <modules/lif/gui/wizardlifconfigpage.h>
#endif

#ifdef BC_MOTOR
#include <modules/motor/gui/wizardmotorscanconfigpage.h>
#endif

ExperimentWizard::ExperimentWizard(Experiment *exp, std::map<QString,QString> hw, QWidget *parent) :
    QWizard(parent), p_experiment(exp)
{
    setWindowTitle(QString("Experiment Setup"));

    auto startPage = new WizardStartPage(this);
    setPage(StartPage,startPage);

    auto loScanConfigPage = new WizardLoScanConfigPage(this);
    setPage(LoScanPage,loScanConfigPage);

    auto drScanConfigPage = new WizardDrScanConfigPage(this);
    setPage(DrScanPage,drScanConfigPage);

    auto rfConfigPage = new WizardRfConfigPage(this);
    setPage(RfConfigPage,rfConfigPage);

    auto chirpConfigPage = new WizardChirpConfigPage(this);
    setPage(ChirpConfigPage,chirpConfigPage);

    auto digitizerConfigPage = new WizardDigitizerConfigPage(this);
    setPage(DigitizerConfigPage,digitizerConfigPage);

    if(hw.find(BC::Key::PGen::key) != hw.end())
    {
        auto pulseConfigPage = new WizardPulseConfigPage(this);
        setPage(PulseConfigPage,pulseConfigPage);
        d_optionalPages.append(PulseConfigPage);
    }

    if(hw.find(BC::Key::IOB::ioboard) != hw.end())
    {
        auto iobConfigPage = new WizardIOBoardConfigPage(this);
        setPage(IOBoardConfigPage,iobConfigPage);
        d_optionalPages.append(IOBoardConfigPage);
    }

    auto validationPage = new WizardValidationPage(this);
    setPage(ValidationPage,validationPage);

    auto summaryPage = new WizardSummaryPage(this);
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

    setSizePolicy(QSizePolicy::Preferred,QSizePolicy::Preferred);
}

ExperimentWizard::~ExperimentWizard()
{ 
}

void ExperimentWizard::setValidationKeys(const std::map<QString, QStringList> &m)
{
    static_cast<WizardValidationPage*>(page(ValidationPage))->setValidationKeys(m);
}

ExperimentWizard::Page ExperimentWizard::nextOptionalPage()
{
    for(auto id : d_optionalPages)
    {
        if(!hasVisitedPage(id))
            return id;
    }

    return ValidationPage;
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
    for(auto id : pageIds())
    {
        auto p = dynamic_cast<ExperimentWizardPage*>(page(id));
        if(p)
            p->discardChanges();
    }

    QDialog::reject();
}
