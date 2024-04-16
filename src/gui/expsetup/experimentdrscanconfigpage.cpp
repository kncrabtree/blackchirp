#include "experimentdrscanconfigpage.h"

using namespace BC::Key::WizDR;

#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QLabel>
#include <QFormLayout>
#include <QGroupBox>
#include <QVBoxLayout>

#include <data/experiment/ftmwconfigtypes.h>

ExperimentDRScanConfigPage::ExperimentDRScanConfigPage(Experiment *exp, QWidget *parent)
    : ExperimentConfigPage(key,title,exp,parent)
{
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


    auto gb = new QGroupBox(QString("DR Scan Settings"));
    auto l = new QFormLayout;

    l->addRow(QString("Start"),p_startBox);
    l->addRow(QString("Step Size"),p_stepSizeBox);
    l->addRow(QString("Num Steps"),p_numStepsBox);
    l->addRow(QString("End"),p_endBox);
    l->addRow(QString("Shots Per Step"),p_shotsBox);
    gb->setLayout(l);

    for(int i=0; i<l->rowCount(); i++)
    {
        auto lbl = static_cast<QLabel*>(l->itemAt(i,QFormLayout::LabelRole)->widget());
        lbl->setAlignment(Qt::AlignRight|Qt::AlignVCenter);
    }

    auto vbl = new QVBoxLayout;
    vbl->addWidget(gb,0);
    vbl->addSpacerItem(new QSpacerItem(0,0));
    setLayout(vbl);

    auto vc = static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged);
    auto dvc = static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged);

    if(exp->d_number > 0)
    {
        auto ftc = dynamic_cast<FtmwConfigDRScan*>(exp->ftmwConfig());
        if(ftc)
        {
            p_startBox->setValue(ftc->d_start);
            p_stepSizeBox->setValue(ftc->d_step);
            p_numStepsBox->setValue(ftc->d_numSteps);
            p_shotsBox->setValue(ftc->d_rfConfig.d_shotsPerClockConfig);
        }
    }

    connect(p_startBox,dvc,this,&ExperimentDRScanConfigPage::updateEndBox);
    connect(p_stepSizeBox,dvc,this,&ExperimentDRScanConfigPage::updateEndBox);
    connect(p_numStepsBox,vc,this,&ExperimentDRScanConfigPage::updateEndBox);

    registerGetter(BC::Key::WizDR::start,p_startBox,&QDoubleSpinBox::value);
    registerGetter(BC::Key::WizDR::step,p_stepSizeBox,&QDoubleSpinBox::value);
    registerGetter(BC::Key::WizDR::numSteps,p_numStepsBox,&QSpinBox::value);
    registerGetter(BC::Key::WizDR::shots,p_shotsBox,&QSpinBox::value);

    updateEndBox();
}

void ExperimentDRScanConfigPage::initialize()
{
    if(!isEnabled() || !p_exp->ftmwConfig())
        return;

    auto const &rfc = p_exp->ftmwConfig()->d_rfConfig;

    //Get DR hardware
    auto r = rfc.clockRange(RfConfig::DRClock);

     p_startBox->blockSignals(true);
     p_stepSizeBox->blockSignals(true);

     p_startBox->setRange(r.first,r.second);
     p_stepSizeBox->setRange(-r.second,r.second);

     p_startBox->blockSignals(false);
     p_stepSizeBox->blockSignals(false);
}

bool ExperimentDRScanConfigPage::validate()
{
    if(!isEnabled() || !p_exp->ftmwConfig())
        return true;

    auto range = p_exp->ftmwConfig()->d_rfConfig.clockRange(RfConfig::DRClock);
    bool out = true;
    if(p_startBox->value() < range.first || p_startBox->value() > range.second)
    {
        emit error("DR Scan starting frequency out of range.");
        out = false;
    }
    if(p_endBox->value() < range.first || p_endBox->value() > range.second)
    {
        emit error("DR Scan ending frequency out of range.");
        out = false;
    }

    return out;
}

void ExperimentDRScanConfigPage::apply()
{
    auto ftc = dynamic_cast<FtmwConfigDRScan*>(p_exp->ftmwConfig());
    if(!ftc)
        return;
    auto &rfc = p_exp->ftmwConfig()->d_rfConfig;

    rfc.clearClockSteps();
    for(int i=0; i<p_numStepsBox->value(); i++)
    {
        double thisFreq = p_startBox->value() + static_cast<double>(i)*p_stepSizeBox->value();
        rfc.addDrScanClockStep(thisFreq);
    }

    rfc.d_targetSweeps = 1;
    rfc.d_shotsPerClockConfig = p_shotsBox->value();
    ftc->d_start = p_startBox->value();
    ftc->d_step = p_stepSizeBox->value();
    ftc->d_numSteps = p_numStepsBox->value();
}

void ExperimentDRScanConfigPage::updateEndBox()
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
