#include "wizardstartpage.h"

#include <QFormLayout>
#include <QCheckBox>
#include <QSpinBox>

#include "experimentwizard.h"

WizardStartPage::WizardStartPage(QWidget *parent) :
    ExperimentWizardPage(parent)
{
    setTitle(QString("Configure Experiment"));
    setSubTitle(QString("Choose which type(s) of experiment you wish to perform."));

    QFormLayout *fl = new QFormLayout(this);

    p_ftmw = new QCheckBox(this);

    p_auxDataIntervalBox = new QSpinBox(this);
    p_auxDataIntervalBox->setRange(5,__INT_MAX__);
    p_auxDataIntervalBox->setValue(300);
    p_auxDataIntervalBox->setSingleStep(300);
    p_auxDataIntervalBox->setSuffix(QString(" s"));
    p_auxDataIntervalBox->setToolTip(QString("Interval for aux data readings (e.g., flows, pressure, etc.)"));

    p_snapshotBox = new QSpinBox(this);
    p_snapshotBox->setRange(1<<8,(1<<30)-1);
    p_snapshotBox->setValue(20000);
    p_snapshotBox->setSingleStep(5000);
    p_snapshotBox->setPrefix(QString("every "));
    p_snapshotBox->setSuffix(QString(" shots"));
    p_snapshotBox->setToolTip(QString("Interval for taking experiment snapshots (i.e., autosaving)."));


    registerField(QString("auxDataInterval"),p_auxDataIntervalBox);
    registerField(QString("snapshotInterval"),p_snapshotBox);

    fl->addRow(QString("FTMW"),p_ftmw);
    connect(p_ftmw,&QCheckBox::toggled,this,&WizardStartPage::completeChanged);
    registerField(QString("ftmw"),p_ftmw);

#ifndef BC_LIF

    p_ftmw->setChecked(true);
#ifndef BC_MOTOR
    p_ftmw->setEnabled(false);
#endif
#else
    p_lif = new QCheckBox(this);
    fl->addRow(QString("LIF"),p_lif);
    connect(p_lif,&QCheckBox::toggled,this,&WizardStartPage::completeChanged);
    registerField(QString("lif"),p_lif);
#endif

#ifdef BC_MOTOR
    p_motor = new QCheckBox(this);
    fl->addRow(QString("Motor Scan"),p_motor);
    connect(p_motor,&QCheckBox::toggled,this,&WizardStartPage::completeChanged);
    connect(p_motor,&QCheckBox::toggled,[=](bool ch){
        p_ftmw->setDisabled(ch);
        if(ch)
            p_ftmw->setChecked(false);
#ifdef BC_LIF
        p_lif->setDisabled(ch);
        if(ch)
            p_lif->setChecked(false);
#endif
    });
#endif
    fl->addRow(QString("Aux Data Interval"),p_auxDataIntervalBox);
    fl->addRow(QString("Snapshot Interval"),p_snapshotBox);

    setLayout(fl);
}

WizardStartPage::~WizardStartPage()
{
}


int WizardStartPage::nextId() const
{
#ifdef BC_MOTOR
    if(p_motor->isChecked())
        return ExperimentWizard::MotorScanConfigPage;
#endif

#ifdef BC_LIF
    if(p_lif->isChecked())
        return ExperimentWizard::LifConfigPage;
    else
        return ExperimentWizard::RfConfigPage;
#endif

    return ExperimentWizard::RfConfigPage;
}

bool WizardStartPage::isComplete() const
{
    bool out = p_ftmw->isChecked();
#ifdef BC_LIF
    out = out || p_lif->isChecked();
#endif

#ifdef BC_MOTOR
    out = out || p_motor->isChecked();
#endif

    return out;
}

void WizardStartPage::initializePage()
{
    auto e = getExperiment();

#ifdef BC_LIF
    p_ftmw->setChecked(e.ftmwConfig().isEnabled());
    p_lif->setChecked(e.lifConfig().isEnabled());
#endif

#ifdef BC_MOTOR
    p_motor->setChecked(e.motorScan().isEnabled());
#endif

    p_snapshotBox->setValue(e.autoSaveShots());
    p_auxDataIntervalBox->setValue(e.timeDataInterval());

}

bool WizardStartPage::ftmwEnabled() const
{
    return p_ftmw->isChecked();
}

bool WizardStartPage::lifEnabled() const
{
#ifdef BC_LIF
    return p_lif->isChecked();
#endif

    return false;
}

bool WizardStartPage::motorEnabled() const
{
#ifdef BC_MOTOR
    return p_motor->isChecked();
#endif

    return false;
}

int WizardStartPage::auxDataInterval() const
{
    return p_auxDataIntervalBox->value();
}

int WizardStartPage::snapshotInterval() const
{
    return p_snapshotBox->value();
}


bool WizardStartPage::validatePage()
{
    ///TODO: In the future, allow user to choose old experiment to repeat!
    ///Be sure to give user the options to use current pulse settings.
    /// Allow changing flow settings?
     auto e = getExperiment();

     e.setFtmwEnabled(p_ftmw->isChecked());

#ifdef BC_LIF
     e.setLifEnabled(p_lif->isChecked());
#endif

#ifdef BC_MOTOR
     e.setMotorEnabled(p_motor->isChecked());
#endif

     emit experimentUpdate(e);
     return true;

}
