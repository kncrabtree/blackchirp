#include "wizarddrscanconfigpage.h"

#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QFormLayout>

WizardDrScanConfigPage::WizardDrScanConfigPage(QWidget *parent) : ExperimentWizardPage(parent)
{
    setTitle(QString("Configure DR Scan"));
    setSubTitle(QString("Hover over the various fields for more information."));

    p_startBox = new QDoubleSpinBox;
    p_startBox->setDecimals(6);
    p_startBox->setSuffix(QString(" MHz"));
    p_startBox->setSingleStep(1000.0);
    p_startBox->setRange(0,1e9);
    p_startBox->setToolTip(QString("Starting DR frequency."));
    p_startBox->setKeyboardTracking(false);


    p_stepSizeBox = new QDoubleSpinBox;
    p_stepSizeBox->setDecimals(6);
    p_stepSizeBox->setSuffix(QString(" MHz"));
    p_stepSizeBox->setSingleStep(0.1);
    p_stepSizeBox->setRange(-1e9,1e9);
    p_stepSizeBox->setToolTip(QString("DR step size (can be negative)."));
    p_stepSizeBox->setKeyboardTracking(false);


    p_numStepsBox = new QSpinBox;
    p_numStepsBox->setRange(2,__INT_MAX__);
    p_numStepsBox->setToolTip(QString("Number of steps to take."));
    p_numStepsBox->setKeyboardTracking(false);


    p_endBox = new QDoubleSpinBox;
    p_endBox->setDecimals(6);
    p_endBox->setRange(0,1e9);
    p_endBox->setSuffix(QString(" MHz"));
    p_endBox->setToolTip(QString("Ending DR frequency. Set automatically."));
    p_endBox->setEnabled(false);

    p_shotsBox = new QSpinBox;
    p_shotsBox->setRange(0,__INT_MAX__);
    p_shotsBox->setSingleStep(100);
    p_shotsBox->setToolTip(QString("Number of shots to acquire at each DR point."));


    auto l = new QFormLayout;

    l->addRow(QString("Start"),p_startBox);
    l->addRow(QString("Step Size"),p_stepSizeBox);
    l->addRow(QString("Num Steps"),p_numStepsBox);
    l->addRow(QString("End"),p_endBox);
    l->addRow(QString("Shots Per Step"),p_shotsBox);

    setLayout(l);

    auto vc = static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged);
    auto dvc = static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged);

    connect(p_startBox,dvc,this,&WizardDrScanConfigPage::updateEndBox);
    connect(p_stepSizeBox,dvc,this,&WizardDrScanConfigPage::updateEndBox);
    connect(p_numStepsBox,vc,this,&WizardDrScanConfigPage::updateEndBox);

    loadFromSettings();

}


void WizardDrScanConfigPage::initializePage()
{
    auto e = getExperiment();
    d_rfConfig = e->ftmwConfig().rfConfig();

    //Get DR hardware
    auto drClock = d_rfConfig.clockHardware(BlackChirp::DRClock);
    if(drClock.isEmpty())
        return;

     QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
     s.beginGroup(drClock);
     s.beginGroup(s.value(QString("subKey"),QString("fixed")).toString());

     double minFreq = s.value(QString("minFreqMHz"),0.0).toDouble();
     double maxFreq = s.value(QString("maxFreqMHz"),1e7).toDouble();

     s.endGroup();
     s.endGroup();

     auto clocks = d_rfConfig.getClocks();
     auto drc = clocks.value(BlackChirp::DRClock);
     if(drc.op == RfConfig::Multiply)
     {
         minFreq*=drc.factor;
         maxFreq*=drc.factor;
     }
     else
     {
         minFreq/=drc.factor;
         maxFreq/=drc.factor;
     }

     p_startBox->blockSignals(true);
     p_stepSizeBox->blockSignals(true);

     p_startBox->setRange(minFreq,maxFreq);
     p_stepSizeBox->setRange(-maxFreq,maxFreq);

     p_startBox->blockSignals(false);
     p_stepSizeBox->blockSignals(false);

     updateEndBox();

}

bool WizardDrScanConfigPage::validatePage()
{
    auto e = getExperiment();

    auto c = d_rfConfig.getClocks();
    d_rfConfig.clearClockSteps();
    for(int i=0; i<p_numStepsBox->value(); i++)
    {
        double thisFreq = p_startBox->value() + static_cast<double>(i)*p_stepSizeBox->value();
        d_rfConfig.addDrScanClockStep(thisFreq);
    }

    d_rfConfig.setTargetSweeps(1);
    d_rfConfig.setShotsPerClockStep(p_shotsBox->value());

    e->setRfConfig(d_rfConfig);

    saveToSettings();
    return true;


}

bool WizardDrScanConfigPage::isComplete() const
{
    return true;
}

int WizardDrScanConfigPage::nextId() const
{
    return ExperimentWizard::ChirpConfigPage;
}

void WizardDrScanConfigPage::updateEndBox()
{
    double end = p_startBox->value() + static_cast<double>(p_numStepsBox->value()-1)*p_stepSizeBox->value();
    if(end < p_endBox->minimum())
    {
        int newNumSteps = static_cast<int>(floor(fabs((p_startBox->value()-p_endBox->minimum())/p_stepSizeBox->value()))) - 1;
        p_numStepsBox->setValue(newNumSteps);
        end = p_startBox->value() + static_cast<double>(p_numStepsBox->value())*p_stepSizeBox->value();
    }
    else if (end > p_endBox->maximum())
    {
        int newNumSteps = static_cast<int>(floor(fabs((p_startBox->value()-p_endBox->maximum())/p_stepSizeBox->value()))) - 1;
        p_numStepsBox->setValue(newNumSteps);
        end = p_startBox->value() + static_cast<double>(p_numStepsBox->value()-1)*p_stepSizeBox->value();
    }
    p_endBox->setValue(end);
}

void WizardDrScanConfigPage::saveToSettings() const
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(QString("lastDrScan"));

    s.setValue(QString("startFreqMHz"),p_startBox->value());
    s.setValue(QString("stepSizeMHz"),p_stepSizeBox->value());
    s.setValue(QString("numSteps"),p_numStepsBox->value());
    s.setValue(QString("numShots"),p_shotsBox->value());

    s.endGroup();
    s.sync();
}

void WizardDrScanConfigPage::loadFromSettings()
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(QString("lastDrScan"));

    p_startBox->blockSignals(true);
    p_stepSizeBox->blockSignals(true);
    p_numStepsBox->blockSignals(true);

    p_startBox->setValue(s.value(QString("startFreqMHz"),p_startBox->minimum()).toDouble());
    p_stepSizeBox->setValue(s.value(QString("stepSizeMHz"),1.0).toDouble());
    p_numStepsBox->setValue(s.value(QString("numSteps"),100).toInt());
    p_shotsBox->setValue(s.value(QString("numShots"),100).toInt());

    p_startBox->blockSignals(false);
    p_stepSizeBox->blockSignals(false);
    p_numStepsBox->blockSignals(false);

    updateEndBox();

    s.endGroup();
}
