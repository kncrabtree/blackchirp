#include "wizardchirpconfigpage.h"

#include <QVBoxLayout>
#include <QDialogButtonBox>
#include <QSpinBox>

#include "chirpconfigwidget.h"
#include "experimentwizard.h"

WizardChirpConfigPage::WizardChirpConfigPage(QWidget *parent) :
    QWizardPage(parent)
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


int WizardChirpConfigPage::nextId() const
{
    return ExperimentWizard::FtmwConfigPage;
}

bool WizardChirpConfigPage::isComplete() const
{
    return p_ccw->getRfConfig().isValid();
}

RfConfig WizardChirpConfigPage::getRfConfig() const
{
    return p_ccw->getRfConfig();
}
