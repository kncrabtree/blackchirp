#include "wizardpulseconfigpage.h"

#include <QVBoxLayout>
#include <QMessageBox>

#include <gui/widget/pulseconfigwidget.h>

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
        if(e->pGenConfig())
            p_pcw->setFromConfig(*e->pGenConfig());

        p_pcw->configureForWizard();
        emit completeChanged();
        d_firstInitialization = false;
    }


#ifdef BC_LIF
    if(e->lifEnabled())
        p_pcw->configureLif(*e->lifConfig());
#endif

    if(e->ftmwEnabled())
        p_pcw->configureFtmw(*e->ftmwConfig());
}

int WizardPulseConfigPage::nextId() const
{
    return static_cast<ExperimentWizard*>(wizard())->nextOptionalPage();
}


bool WizardPulseConfigPage::validatePage()
{
    if(!p_pcw->d_wizardOk)
        return false;

    auto cfg = p_pcw->getConfig();
    if(!cfg.d_pulseEnabled)
    {
        int ret = QMessageBox::warning(this,QString("Pulsing Disabled"),"You have disabled the pulse generator, and therefore no pulses will be generated. Do you wish to proceed with this setting?",QMessageBox::Yes|QMessageBox::No,QMessageBox::No);
        if(ret == QMessageBox::No)
            return false;
    }

    auto e = getExperiment();
    e->setPulseGenConfig(p_pcw->getConfig());
    

    return true;
}

bool WizardPulseConfigPage::isComplete() const
{
    return p_pcw->d_wizardOk;
}
