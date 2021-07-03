#include "wizarddrscanconfigpage.h"

#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QLabel>
#include <QFormLayout>

#include <hardware/core/clock/clock.h>

WizardDrScanConfigPage::WizardDrScanConfigPage(QWidget *parent) : ExperimentWizardPage(BC::Key::WizDR::key,parent)
{
    setTitle(QString("Configure DR Scan"));
    setSubTitle(QString("Hover over the various fields for more information."));

    p_startBox = new QDoubleSpinBox;
    p_startBox->setDecimals(6);
    p_startBox->setSuffix(QString(" MHz"));
    p_startBox->setSingleStep(1000.0);
    p_startBox->setRange(0,1e9);
    p_startBox->setValue(get<double>(BC::Key::WizDR::start,p_startBox->minimum()));
    p_startBox->setToolTip(QString("Starting DR frequency."));
    p_startBox->setKeyboardTracking(false);


    p_stepSizeBox = new QDoubleSpinBox;
    p_stepSizeBox->setDecimals(6);
    p_stepSizeBox->setSuffix(QString(" MHz"));
    p_stepSizeBox->setSingleStep(0.1);
    p_stepSizeBox->setRange(-1e9,1e9);
    p_stepSizeBox->setValue(get<double>(BC::Key::WizDR::step,1.0));
    p_stepSizeBox->setToolTip(QString("DR step size (can be negative)."));
    p_stepSizeBox->setKeyboardTracking(false);


    p_numStepsBox = new QSpinBox;
    p_numStepsBox->setRange(2,__INT_MAX__);
    p_numStepsBox->setToolTip(QString("Number of steps to take."));
    p_numStepsBox->setValue(get<int>(BC::Key::WizDR::numSteps,100));
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
    p_shotsBox->setValue(get<int>(BC::Key::WizDR::shots,100));
    p_shotsBox->setToolTip(QString("Number of shots to acquire at each DR point."));


    auto l = new QFormLayout;

    l->addRow(QString("Start"),p_startBox);
    l->addRow(QString("Step Size"),p_stepSizeBox);
    l->addRow(QString("Num Steps"),p_numStepsBox);
    l->addRow(QString("End"),p_endBox);
    l->addRow(QString("Shots Per Step"),p_shotsBox);

    for(int i=0; i<l->rowCount(); i++)
    {
        auto lbl = static_cast<QLabel*>(l->itemAt(i,QFormLayout::LabelRole)->widget());
        lbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Expanding);
        lbl->setAlignment(Qt::AlignRight|Qt::AlignVCenter);
    }

    setLayout(l);

    auto vc = static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged);
    auto dvc = static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged);

    connect(p_startBox,dvc,this,&WizardDrScanConfigPage::updateEndBox);
    connect(p_stepSizeBox,dvc,this,&WizardDrScanConfigPage::updateEndBox);
    connect(p_numStepsBox,vc,this,&WizardDrScanConfigPage::updateEndBox);

    registerGetter(BC::Key::WizDR::start,p_startBox,&QDoubleSpinBox::value);
    registerGetter(BC::Key::WizDR::step,p_stepSizeBox,&QDoubleSpinBox::value);
    registerGetter(BC::Key::WizDR::numSteps,p_numStepsBox,&QSpinBox::value);
    registerGetter(BC::Key::WizDR::shots,p_shotsBox,&QSpinBox::value);

}


void WizardDrScanConfigPage::initializePage()
{
    auto e = getExperiment();
    d_rfConfig = e->d_ftmwCfg.rfConfig();

    //Get DR hardware
    auto drClock = d_rfConfig.clockHardware(BlackChirp::DRClock);
    if(drClock.isEmpty())
        return;

     SettingsStorage s(drClock,Hardware);
     double minFreq = s.get<double>(BC::Key::Clock::minFreq,0.0);
     double maxFreq = s.get<double>(BC::Key::Clock::maxFreq,1e7);

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
