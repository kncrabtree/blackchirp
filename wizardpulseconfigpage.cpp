#include "wizardpulseconfigpage.h"

#include <QVBoxLayout>
#include <QMessageBox>

#include "pulseconfigwidget.h"
#include "experimentwizard.h"

WizardPulseConfigPage::WizardPulseConfigPage(QWidget *parent) :
    QWizardPage(parent)
{
    setTitle("Configure Pulses");
    setSubTitle("Some settings will be made automatically (e.g., LIF delays).");

    QVBoxLayout *vbl = new QVBoxLayout();

    p_pcw = new PulseConfigWidget(this);
    p_pcw->makeInternalConnections();
    vbl->addWidget(p_pcw);

    setLayout(vbl);
}

WizardPulseConfigPage::~WizardPulseConfigPage()
{

}

void WizardPulseConfigPage::setConfig(const PulseGenConfig c)
{
    p_pcw->newConfig(c);
}

PulseGenConfig WizardPulseConfigPage::getConfig() const
{
    return p_pcw->getConfig();
}

void WizardPulseConfigPage::initializePage()
{
#ifndef BC_NO_LIF
    if(field(QString("lif")).toBool())
        p_pcw->configureLif(field(QString("delayStart")).toDouble());
#endif

    if(field(QString("ftmw")).toBool())
        p_pcw->configureChirp();
}

int WizardPulseConfigPage::nextId() const
{
    return ExperimentWizard::ValidationPage;
}


bool WizardPulseConfigPage::validatePage()
{
    if(p_pcw->getConfig().at(BC_PGEN_GASCHANNEL).enabled == false)
    {
        QMessageBox::StandardButton ret = QMessageBox::question(this,QString("Gas Pulse Disabled"),QString("Are you sure you want to run this experiment with no gas pulses?"));
        if(ret == QMessageBox::Yes)
            return true;
        else
            return false;
    }

    return true;
}
