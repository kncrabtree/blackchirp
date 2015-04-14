#include "wizardstartpage.h"

#include <QVBoxLayout>
#include "experimentwizard.h"

WizardStartPage::WizardStartPage(QWidget *parent) :
    QWizardPage(parent)
{
    setTitle(QString("Configure Experiment"));
    setSubTitle(QString("Choose which type(s) of experiment you wish to perform."));

    QVBoxLayout *vbl = new QVBoxLayout(this);

    p_ftmw = new QCheckBox(QString("FTMW"),this);
    p_lif = new QCheckBox(QString("LIF"),this);
    p_lif->setEnabled(false);

    connect(p_ftmw,&QCheckBox::toggled,this,&WizardStartPage::completeChanged);
    connect(p_lif,&QCheckBox::toggled,this,&WizardStartPage::completeChanged);

    registerField(QString("lif"),p_lif);

    vbl->addWidget(p_ftmw);
    vbl->addWidget(p_lif);

    setLayout(vbl);
}

WizardStartPage::~WizardStartPage()
{
}


int WizardStartPage::nextId() const
{
    if(p_lif->isChecked())
        return ExperimentWizard::LifConfigPage;
    else
        return ExperimentWizard::ChirpConfigPage;
}

bool WizardStartPage::isComplete() const
{
    return (p_ftmw->isChecked() || p_lif->isChecked());
}

bool WizardStartPage::ftmwEnabled() const
{
    return p_ftmw->isChecked();
}

bool WizardStartPage::lifEnabled() const
{
    return p_lif->isChecked();
}
