#include "wizardstartpage.h"

#include <QFormLayout>
#include <QCheckBox>
#include <QSpinBox>

#include "experimentwizard.h"

WizardStartPage::WizardStartPage(QWidget *parent) :
    QWizardPage(parent)
{
    setTitle(QString("Configure Experiment"));
    setSubTitle(QString("Choose which type(s) of experiment you wish to perform."));

    QFormLayout *fl = new QFormLayout(this);

    p_ftmw = new QCheckBox(this);
    p_lif = new QCheckBox(this);

    p_auxDataIntervalBox = new QSpinBox(this);
    p_auxDataIntervalBox->setRange(5,__INT_MAX__);
    p_auxDataIntervalBox->setValue(300);
    p_auxDataIntervalBox->setSuffix(QString(" s"));
    p_auxDataIntervalBox->setToolTip(QString("Interval for aux data readings (e.g., flows, pressure, etc.)"));

    connect(p_ftmw,&QCheckBox::toggled,this,&WizardStartPage::completeChanged);
    connect(p_lif,&QCheckBox::toggled,this,&WizardStartPage::completeChanged);

    registerField(QString("lif"),p_lif);
    registerField(QString("ftmw"),p_ftmw);
    registerField(QString("auxDataInterval"),p_auxDataIntervalBox);

    fl->addRow(QString("FTMW"),p_ftmw);
    fl->addRow(QString("LIF"),p_lif);
    fl->addRow(QString("Aux Data Interval"),p_auxDataIntervalBox);

    setLayout(fl);
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

int WizardStartPage::auxDataInterval() const
{
    return p_auxDataIntervalBox->value();
}
