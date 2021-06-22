#include "wizardpulseconfigpage.h"

#include <QVBoxLayout>
#include <QMessageBox>

#include <src/gui/widget/pulseconfigwidget.h>

WizardPulseConfigPage::WizardPulseConfigPage(QWidget *parent) :
    ExperimentWizardPage(BC::Key::WizPulse::key,parent), d_firstInitialization(true)
{
    setTitle("Configure Pulses");
    setSubTitle("Some settings may be made automatically (e.g., LIF delays).");

    QVBoxLayout *vbl = new QVBoxLayout();

    p_pcw = new PulseConfigWidget(this);
    vbl->addWidget(p_pcw);

    setLayout(vbl);
}

WizardPulseConfigPage::~WizardPulseConfigPage()
{

}

void WizardPulseConfigPage::initializePage()
{
    auto e = getExperiment();
    if(d_firstInitialization)
    {
        p_pcw->setFromConfig(e->pGenConfig());
        p_pcw->configureForWizard();
        d_firstInitialization = false;
    }


#ifdef BC_LIF
    if(e->lifConfig().isEnabled())
        p_pcw->configureLif(e->lifConfig());
#endif

    if(e->ftmwConfig().isEnabled())
        p_pcw->configureFtmw(e->ftmwConfig());
}

int WizardPulseConfigPage::nextId() const
{
    return ExperimentWizard::ValidationPage;
}


bool WizardPulseConfigPage::validatePage()
{
    auto e = getExperiment();
    e->setPulseGenConfig(p_pcw->getConfig());
    

    return true;
}
