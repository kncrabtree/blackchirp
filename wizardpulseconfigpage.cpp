#include "wizardpulseconfigpage.h"

#include <QVBoxLayout>
#include <QMessageBox>

#include "pulseconfigwidget.h"

WizardPulseConfigPage::WizardPulseConfigPage(QWidget *parent) :
    ExperimentWizardPage(parent)
{
    setTitle("Configure Pulses");
    setSubTitle("Some settings may be made automatically (e.g., LIF delays).");

    QVBoxLayout *vbl = new QVBoxLayout();

    p_pcw = new PulseConfigWidget(this);
    p_pcw->makeInternalConnections();
    vbl->addWidget(p_pcw);

    setLayout(vbl);
}

WizardPulseConfigPage::~WizardPulseConfigPage()
{

}

void WizardPulseConfigPage::initializePage()
{
    auto e = getExperiment();
    p_pcw->setFromConfig(e.pGenConfig());

#ifdef BC_LIF
    ///TODO: can set this directly from LifConfig now instead of using field
    /// Also need to work with new mechanism for special channels
    if(e.lifConfig().isEnabled())
        p_pcw->configureLif(field(QString("delayStart")).toDouble());
#endif

    if(e.ftmwConfig().isEnabled())
        p_pcw->configureChirp();
}

int WizardPulseConfigPage::nextId() const
{
    return ExperimentWizard::ValidationPage;
}


bool WizardPulseConfigPage::validatePage()
{
    auto e = getExperiment();
    e.setPulseGenConfig(p_pcw->getConfig());
    emit experimentUpdate(e);
    return true;
}
