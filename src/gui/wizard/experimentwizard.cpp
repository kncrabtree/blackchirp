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

ExperimentWizard::ExperimentWizard(Experiment *exp, const std::map<QString, QString> &hw, QWidget *parent) :
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
    p_lifConfigPage = new WizardLifConfigPage(this);
    setPage(LifConfigPage,p_lifConfigPage);
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

#ifdef BC_LIF
LifControlWidget *ExperimentWizard::lifControlWidget()
{
    return p_lifConfigPage->controlWidget();
}
#endif

QSize ExperimentWizard::sizeHint() const
{
    return {1000,700};
}


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
