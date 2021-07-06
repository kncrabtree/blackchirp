#include "wizardchirpconfigpage.h"

#include <QVBoxLayout>
#include <QDialogButtonBox>
#include <QSpinBox>

#include <gui/widget/chirpconfigwidget.h>
#include <gui/wizard/experimentwizard.h>

WizardChirpConfigPage::WizardChirpConfigPage(QWidget *parent) :
    ExperimentWizardPage(BC::Key::WizChirp::key,parent)
{
    setTitle(QString("Configure FTMW Chirp"));

    p_ccw = new ChirpConfigWidget(this);
    connect(p_ccw,&ChirpConfigWidget::chirpConfigChanged,this,&WizardChirpConfigPage::completeChanged);
    registerField(QString("numChirps"),p_ccw->numChirpsBox());

    QVBoxLayout *vbl = new QVBoxLayout(this);
    vbl->addWidget(p_ccw);

    setLayout(vbl);

}

WizardChirpConfigPage::~WizardChirpConfigPage()
{

}

void WizardChirpConfigPage::initializePage()
{
    //get rfConfig
    auto e = getExperiment();
    p_ccw->setRfConfig(e->ftmwConfig()->d_rfConfig);
    p_ccw->updateChirpPlot();

}


int WizardChirpConfigPage::nextId() const
{
    return ExperimentWizard::DigitizerConfigPage;
}


bool WizardChirpConfigPage::validatePage()
{
    ///TODO: Smarter validation?
    auto e = getExperiment();
    auto rfc = p_ccw->getRfConfig();
    e->ftmwConfig()->d_rfConfig = rfc;

    return true;
}


bool WizardChirpConfigPage::isComplete() const
{
    auto l = p_ccw->getRfConfig().getChirpConfig().chirpList();

    if(l.isEmpty())
        return false;

    for(int i=0; i<l.size(); i++)
    {
        if(l.at(i).isEmpty())
            return false;
    }

    return true;
}
