#include "wizardloscanconfigpage.h"

#include <QGroupBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QGridLayout>
#include <QLabel>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QCheckBox>

#include <hardware/core/clock/clock.h>
#include <data/experiment/ftmwconfigtypes.h>

WizardLoScanConfigPage::WizardLoScanConfigPage(QWidget *parent) :
    ExperimentWizardPage(BC::Key::WizLoScan::key,parent)
{
    setTitle(QString("Configure LO Scan"));
    setSubTitle(QString("Hover over the various fields for more information."));

    p_upBox = new QGroupBox("Upconversion LO");

    p_upStartBox = new QDoubleSpinBox;
    p_upStartBox->setDecimals(6);
    p_upStartBox->setSuffix(QString(" MHz"));
    p_upStartBox->setSingleStep(1000.0);
    p_upStartBox->setRange(0.0,1e9);
    p_upStartBox->setValue(get<double>(BC::Key::WizLoScan::upStart,0.0));
    p_upStartBox->setToolTip(QString("Starting major step LO frequency.\nChanging this value will update the major step size."));
    p_upStartBox->setKeyboardTracking(false);

    p_upEndBox = new QDoubleSpinBox;
    p_upEndBox->setDecimals(6);
    p_upEndBox->setSuffix(QString(" MHz"));
    p_upEndBox->setSingleStep(1000.0);
    p_upEndBox->setRange(0.0,1e9);
    p_upEndBox->setValue(get<double>(BC::Key::WizLoScan::upEnd,1000.0));
    p_upEndBox->setToolTip(QString("Ending LO frequency (including major and minor steps).\nThis is a limit; the frequency will not exceed this value.\nChanging this value will update the major step size."));
    p_upEndBox->setKeyboardTracking(false);

    p_upNumMinorBox = new QSpinBox;
    p_upNumMinorBox->setRange(1,10);
    p_upNumMinorBox->setValue(get<int>(BC::Key::WizLoScan::upNumMinor,1));
    p_upNumMinorBox->setToolTip(QString("Number of minor (small) steps to take per major step.\nThe sign is determined automatically."));

    p_upMinorStepBox = new QDoubleSpinBox;
    p_upMinorStepBox->setDecimals(6);
    p_upMinorStepBox->setSuffix(QString(" MHz"));
    p_upMinorStepBox->setSingleStep(1.0);
    p_upMinorStepBox->setRange(0.0,1e9);
    p_upMinorStepBox->setValue(get<double>(BC::Key::WizLoScan::upMinorStep,0.0));
    p_upMinorStepBox->setToolTip(QString("Minor step size, if number of minor steps > 1.\nMay change the number of major steps."));
    p_upMinorStepBox->setKeyboardTracking(false);

    p_upNumMajorBox = new QSpinBox;
    p_upNumMajorBox->setRange(2,100000);
    p_upNumMajorBox->setValue(get<int>(BC::Key::WizLoScan::upNumMajor,2));
    p_upNumMajorBox->setToolTip(QString("Number of major steps desired.\nChanging this will update the major step size."));

    p_upMajorStepBox = new QDoubleSpinBox;
    p_upMajorStepBox->setDecimals(6);
    p_upMajorStepBox->setSuffix(QString(" MHz"));
    p_upMajorStepBox->setSingleStep(100.0);
    p_upMajorStepBox->setRange(0.0,1e9);
    p_upMajorStepBox->setValue(get<double>(BC::Key::WizLoScan::upMajorStep,1000.0));
    p_upMajorStepBox->setToolTip(QString("Desired major step size.\nChanging this will update the number of major steps."));
    p_upMajorStepBox->setKeyboardTracking(false);

    auto *upgl = new QGridLayout;
    upgl->addWidget(new QLabel("Start"),0,0);
    upgl->addWidget(p_upStartBox,0,1);
    upgl->addWidget(new QLabel("End"),0,2);
    upgl->addWidget(p_upEndBox,0,3);

    upgl->addWidget(new QLabel("Minor Steps/pt"),1,0);
    upgl->addWidget(p_upNumMinorBox,1,1);
    upgl->addWidget(new QLabel("Size"),1,2);
    upgl->addWidget(p_upMinorStepBox,1,3);

    upgl->addWidget(new QLabel("Major Steps"),2,0);
    upgl->addWidget(p_upNumMajorBox,2,1);
    upgl->addWidget(new QLabel("Size"),2,2);
    upgl->addWidget(p_upMajorStepBox,2,3);

    upgl->addItem(new QSpacerItem(0,0,QSizePolicy::Minimum,QSizePolicy::Expanding),3,0);

    p_upBox->setLayout(upgl);



    p_downBox = new QGroupBox("Downconversion LO");

    p_downStartBox = new QDoubleSpinBox;
    p_downStartBox->setDecimals(6);
    p_downStartBox->setSuffix(QString(" MHz"));
    p_downStartBox->setSingleStep(1000.0);
    p_downStartBox->setRange(0.0,1e9);
    p_downStartBox->setValue(get<double>(BC::Key::WizLoScan::downStart,0.0));
    p_downStartBox->setToolTip(QString("Starting major step LO frequency."));
    p_downStartBox->setKeyboardTracking(false);

    p_downEndBox = new QDoubleSpinBox;
    p_downEndBox->setDecimals(6);
    p_downEndBox->setSuffix(QString(" MHz"));
    p_downEndBox->setSingleStep(1000.0);
    p_downEndBox->setRange(0.0,1e9);
    p_downEndBox->setValue(get<double>(BC::Key::WizLoScan::downEnd,1000.0));
    p_downEndBox->setToolTip(QString("Ending LO frequency (including major and minor steps).\nThis is a limit; the frequency will not exceed this value.\nChanging this value will update the major step size."));
    p_downEndBox->setKeyboardTracking(false);

    p_downNumMinorBox = new QSpinBox;
    p_downNumMinorBox->setRange(1,10);
    p_downNumMinorBox->setValue(get<int>(BC::Key::WizLoScan::downNumMinor,1));
    p_downNumMinorBox->setToolTip(QString("Number of minor (small) steps to take per major step.\nThe sign is determined automatically."));

    p_downMinorStepBox = new QDoubleSpinBox;
    p_downMinorStepBox->setDecimals(6);
    p_downMinorStepBox->setSuffix(QString(" MHz"));
    p_downMinorStepBox->setSingleStep(1.0);
    p_downMinorStepBox->setRange(0.0,1e9);
    p_downMinorStepBox->setValue(get<double>(BC::Key::WizLoScan::downMinorStep,0.0));
    p_downMinorStepBox->setToolTip(QString("Minor step size, if number of minor steps > 1.\nMay change the number of major steps."));
    p_downMinorStepBox->setKeyboardTracking(false);

    p_downNumMajorBox = new QSpinBox;
    p_downNumMajorBox->setRange(2,100000);
    p_downNumMajorBox->setValue(get<int>(BC::Key::WizLoScan::downNumMajor,2));
    p_downNumMajorBox->setToolTip(QString("Number of major steps desired.\nChanging this will update the major step size."));

    p_downMajorStepBox = new QDoubleSpinBox;
    p_downMajorStepBox->setDecimals(6);
    p_downMajorStepBox->setSuffix(QString(" MHz"));
    p_downMajorStepBox->setSingleStep(100.0);
    p_downMajorStepBox->setRange(0.0,1e9);
    p_downMajorStepBox->setValue(get<double>(BC::Key::WizLoScan::downMajorStep,1000.0));
    p_downMajorStepBox->setToolTip(QString("Desired major step size.\nChanging this will update the number of major steps."));
    p_downMajorStepBox->setKeyboardTracking(false);

    p_fixedDownLoBox = new QCheckBox(QString("Fixed Frequency"));
    p_fixedDownLoBox->setToolTip(QString("If checked, the downconversion frequency will be set to the start value for all points."));
    p_fixedDownLoBox->setChecked(get<bool>(BC::Key::WizLoScan::downFixed,false));

    p_constantDownOffsetBox = new QCheckBox(QString("Constant Offset"));
    p_constantDownOffsetBox->setToolTip(QString("If checked, the downconversion frequency will maintain a constant difference from the upconversion LO.\nThe difference will be kept at the difference of the start frequencies."));
    p_constantDownOffsetBox->setChecked(get<bool>(BC::Key::WizLoScan::constOffset,false));

    auto *downgl = new QGridLayout;
    downgl->addWidget(new QLabel("Start"),0,0,Qt::AlignRight);
    downgl->addWidget(p_downStartBox,0,1);
    downgl->addWidget(new QLabel("End"),0,2,Qt::AlignRight);
    downgl->addWidget(p_downEndBox,0,3);

    downgl->addWidget(new QLabel("Minor Steps/pt"),1,0,Qt::AlignRight);
    downgl->addWidget(p_downNumMinorBox,1,1);
    downgl->addWidget(new QLabel("Size"),1,2,Qt::AlignRight);
    downgl->addWidget(p_downMinorStepBox,1,3);

    downgl->addWidget(new QLabel("Major Steps"),2,0,Qt::AlignRight);
    downgl->addWidget(p_downNumMajorBox,2,1);
    downgl->addWidget(new QLabel("Size"),2,2,Qt::AlignRight);
    downgl->addWidget(p_downMajorStepBox,2,3);

    downgl->addWidget(p_fixedDownLoBox,3,0,1,2,Qt::AlignLeft);
    downgl->addWidget(p_constantDownOffsetBox,3,2,1,2,Qt::AlignLeft);

    downgl->addItem(new QSpacerItem(0,0,QSizePolicy::Minimum,QSizePolicy::Expanding),4,0);

    p_downBox->setLayout(downgl);

    auto *otherBox = new QGroupBox(QString("Scan Settings"));
    auto *fl = new QFormLayout;

    p_shotsPerStepBox = new QSpinBox;
    p_shotsPerStepBox->setRange(1,__INT_MAX__);
    p_shotsPerStepBox->setSingleStep(1000);
    p_shotsPerStepBox->setValue(get(BC::Key::WizLoScan::shots,1000));
    p_shotsPerStepBox->setToolTip(QString("Number of shots to acquire at each step (major and minor)."));

    auto lbl = new QLabel(QString("Shots/Point"));
    lbl->setAlignment(Qt::AlignRight|Qt::AlignCenter);
    lbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::MinimumExpanding);
    fl->addRow(lbl,p_shotsPerStepBox);


    p_targetSweepsBox = new QSpinBox;
    p_targetSweepsBox->setRange(1,__INT_MAX__);
    p_targetSweepsBox->setValue(get(BC::Key::WizLoScan::sweeps,1));
    p_targetSweepsBox->setToolTip(QString("Number of sweeps through the total LO range.\nExperiment will end when this number is reached."));
    lbl = new QLabel(QString("Target Sweeps"));
    lbl->setAlignment(Qt::AlignRight|Qt::AlignCenter);
    lbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::MinimumExpanding);
    fl->addRow(lbl,p_targetSweepsBox);
    otherBox->setLayout(fl);


    auto *hbl = new QHBoxLayout;
    hbl->addWidget(otherBox);
    hbl->addWidget(p_upBox);
    hbl->addWidget(p_downBox);

    setLayout(hbl);

    auto vc = static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged);
    auto dvc = static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged);

    connect(p_upStartBox,dvc,[=](double v){
        startChanged(RfConfig::UpLO,v);
    });
    connect(p_downStartBox,dvc,[=](double v){
        startChanged(RfConfig::DownLO,v);
    });

    connect(p_upEndBox,dvc,[=](double v){
        endChanged(RfConfig::UpLO,v);
    });
    connect(p_downEndBox,dvc,[=](double v){
        endChanged(RfConfig::DownLO,v);
    });

    connect(p_upMajorStepBox,dvc,[=](double v){
       majorStepSizeChanged(RfConfig::UpLO,v);
    });
    connect(p_downMajorStepBox,dvc,[=](double v){
       majorStepSizeChanged(RfConfig::DownLO,v);
    });

    connect(p_upMinorStepBox,dvc,[=](double v){
       minorStepSizeChanged(RfConfig::UpLO,v);
    });
    connect(p_downMinorStepBox,dvc,[=](double v){
       minorStepSizeChanged(RfConfig::DownLO,v);
    });

    connect(p_upNumMinorBox,vc,[=](int v){
       minorStepChanged(RfConfig::UpLO,v);
    });
    connect(p_downNumMinorBox,vc,[=](int v){
       minorStepChanged(RfConfig::DownLO,v);
    });

    connect(p_upNumMajorBox,vc,[=](int v){
       majorStepChanged(RfConfig::UpLO,v);
    });
    connect(p_downNumMajorBox,vc,[=](int v){
       majorStepChanged(RfConfig::DownLO,v);
    });

    connect(p_constantDownOffsetBox,&QCheckBox::toggled,this,&WizardLoScanConfigPage::constantOffsetChanged);
    connect(p_fixedDownLoBox,&QCheckBox::toggled,this,&WizardLoScanConfigPage::fixedChanged);

    registerGetter(BC::Key::WizLoScan::shots,p_shotsPerStepBox,&QSpinBox::value);
    registerGetter(BC::Key::WizLoScan::sweeps,p_targetSweepsBox,&QSpinBox::value);

    registerGetter(BC::Key::WizLoScan::upStart,p_upStartBox,&QDoubleSpinBox::value);
    registerGetter(BC::Key::WizLoScan::upEnd,p_upEndBox,&QDoubleSpinBox::value);
    registerGetter(BC::Key::WizLoScan::upNumMinor,p_upNumMinorBox,&QSpinBox::value);
    registerGetter(BC::Key::WizLoScan::upMinorStep,p_upMinorStepBox,&QDoubleSpinBox::value);
    registerGetter(BC::Key::WizLoScan::upNumMajor,p_upNumMajorBox,&QSpinBox::value);
    registerGetter(BC::Key::WizLoScan::upMajorStep,p_upMajorStepBox,&QDoubleSpinBox::value);

    registerGetter(BC::Key::WizLoScan::downStart,p_downStartBox,&QDoubleSpinBox::value);
    registerGetter(BC::Key::WizLoScan::downEnd,p_downEndBox,&QDoubleSpinBox::value);
    registerGetter(BC::Key::WizLoScan::downNumMinor,p_downNumMinorBox,&QSpinBox::value);
    registerGetter(BC::Key::WizLoScan::downMinorStep,p_downMinorStepBox,&QDoubleSpinBox::value);
    registerGetter(BC::Key::WizLoScan::downNumMajor,p_downNumMajorBox,&QSpinBox::value);
    registerGetter(BC::Key::WizLoScan::downMajorStep,p_downMajorStepBox,&QDoubleSpinBox::value);

    registerGetter(BC::Key::WizLoScan::downFixed,static_cast<QAbstractButton*>(p_fixedDownLoBox),&QCheckBox::isChecked);
    registerGetter(BC::Key::WizLoScan::constOffset,static_cast<QAbstractButton*>(p_constantDownOffsetBox),&QCheckBox::isChecked);

}

void WizardLoScanConfigPage::initializePage()
{
    auto e = getExperiment();
    auto &rfc = e->ftmwConfig()->d_rfConfig;

    //get LO hardware
    auto upLO = rfc.clockHardware(RfConfig::UpLO);
    auto downLO = rfc.clockHardware(RfConfig::DownLO);

    if(upLO.isEmpty())
        return;

    SettingsStorage s(upLO,Hardware);

    double upMinFreq = s.get<double>(BC::Key::Clock::minFreq,0.0);
    double upMaxFreq = s.get<double>(BC::Key::Clock::maxFreq,1e7);

    double downMinFreq = upMinFreq;
    double downMaxFreq = upMaxFreq;

    if(!rfc.d_commonUpDownLO && upLO != downLO)
    {
        SettingsStorage s2(downLO,Hardware);

        downMinFreq = s2.get<double>(BC::Key::Clock::minFreq,0.0);
        downMaxFreq = s2.get<double>(BC::Key::Clock::maxFreq,1e7);
    }

    auto clocks = rfc.getClocks();
    auto upLoClock = clocks.value(RfConfig::UpLO);
    if(upLoClock.op == RfConfig::Multiply)
    {
        upMinFreq*=upLoClock.factor;
        upMaxFreq*=upLoClock.factor;
    }
    else
    {
        upMinFreq/=upLoClock.factor;
        upMaxFreq/=upLoClock.factor;
    }


    auto downLoClock = clocks.value(RfConfig::DownLO);
    if(downLoClock.op == RfConfig::Multiply)
    {
        downMinFreq*=downLoClock.factor;
        downMaxFreq*=downLoClock.factor;
    }
    else
    {
        downMinFreq/=downLoClock.factor;
        downMaxFreq/=downLoClock.factor;
    }


    p_upStartBox->setRange(upMinFreq,upMaxFreq);
    p_upEndBox->setRange(upMinFreq,upMaxFreq);
    p_upMajorStepBox->setRange(1.0,upMaxFreq-upMinFreq);
    p_upMinorStepBox->setRange(0.0,upMaxFreq-upMinFreq);

    p_downStartBox->setRange(downMinFreq,downMaxFreq);
    p_downEndBox->setRange(downMinFreq,downMaxFreq);
    p_downMajorStepBox->setRange(1.0,downMaxFreq-downMinFreq);
    p_downMinorStepBox->setRange(0.0,downMaxFreq-downMinFreq);

    p_shotsPerStepBox->setValue(e->ftmwConfig()->d_objective);
    p_downBox->setDisabled(rfc.d_commonUpDownLO);

    if(e->d_number > 0)
    {
        p_targetSweepsBox->setValue(rfc.d_targetSweeps);
        auto ftc = dynamic_cast<FtmwConfigLOScan*>(e->ftmwConfig());
        if(ftc)
        {
            p_upStartBox->setValue(ftc->d_upStart);
            p_upEndBox->setValue(ftc->d_upEnd);
            p_upNumMinorBox->setValue(ftc->d_upMin);
            p_upNumMajorBox->setValue(ftc->d_upMaj);
            p_downStartBox->setValue(ftc->d_downStart);
            p_downEndBox->setValue(ftc->d_downEnd);
            p_downNumMinorBox->setValue(ftc->d_downMin);
            p_downNumMajorBox->setValue(ftc->d_downMaj);
        }
    }

    if(rfc.d_commonUpDownLO)
    {
        p_downStartBox->blockSignals(true);
        p_downStartBox->setValue(p_upStartBox->value());
        p_downStartBox->blockSignals(false);

        p_downEndBox->blockSignals(true);
        p_downEndBox->setValue(p_upEndBox->value());
        p_downEndBox->blockSignals(false);

        p_downNumMinorBox->blockSignals(true);
        p_downNumMinorBox->setValue(p_upNumMinorBox->value());
        p_downNumMinorBox->blockSignals(false);

        p_downMinorStepBox->blockSignals(true);
        p_downMinorStepBox->setValue(p_upMinorStepBox->value());
        p_downMinorStepBox->blockSignals(false);

        p_downNumMajorBox->blockSignals(true);
        p_downNumMajorBox->setValue(p_upNumMajorBox->value());
        p_downNumMajorBox->blockSignals(false);

        p_downMajorStepBox->blockSignals(true);
        p_downMajorStepBox->setValue(p_upMajorStepBox->value());
        p_downMajorStepBox->blockSignals(false);
    }
}

bool WizardLoScanConfigPage::validatePage()
{
    auto e = getExperiment();
    auto &rfc = e->ftmwConfig()->d_rfConfig;
    auto ftc = dynamic_cast<FtmwConfigLOScan*>(e->ftmwConfig());

    QVector<double> upLoValues, downLoValues;
    double direction = 1.0;
    double start = p_upStartBox->value();
    double end = p_upEndBox->value();
    int numMinor = p_upNumMinorBox->value();
    int numMajor = p_upNumMajorBox->value();
    double minorSize = p_upMinorStepBox->value();
    double majorStep = p_upMajorStepBox->value();
    if(ftc)
    {
        ftc->d_upStart = start;
        ftc->d_upEnd = end;
        ftc->d_upMaj = numMajor;
        ftc->d_upMin = numMinor;
    }

    if(end < start)
        direction *= -1.0;

    for(int i=0; i<numMajor; i++)
    {
        double thisMajorFreq = start + direction*majorStep*static_cast<double>(i);
        upLoValues << thisMajorFreq;
        for(int j=1; j<numMinor; j++)
            upLoValues << thisMajorFreq + minorSize*direction*static_cast<double>(j);
    }

    double offset = p_downStartBox->value() - start;


    if(rfc.d_commonUpDownLO)
        downLoValues = upLoValues;
    else if(p_fixedDownLoBox->isChecked())
    {
        for(int i=0; i<upLoValues.size(); i++)
            downLoValues << start;
    }
    else if(p_constantDownOffsetBox->isChecked())
    {
        start = p_upStartBox->value() + offset;
        end = p_upEndBox->value() + offset;
        for(int i=0; i<upLoValues.size(); i++)
            downLoValues << upLoValues.at(i) + offset;
    }
    else
    {
        direction = 1.0;
        start = p_downStartBox->value();
        end = p_downEndBox->value();
        numMinor = p_downNumMinorBox->value();
        numMajor = p_downNumMajorBox->value();
        minorSize = p_downMinorStepBox->value();
        majorStep = p_downMajorStepBox->value();
        if(end < start)
            direction *= -1.0;


        for(int i=0; i<numMajor; i++)
        {
            double thisMajorFreq = start + direction*majorStep*static_cast<double>(i);
            downLoValues << thisMajorFreq;
            for(int j=1; j<numMinor; j++)
                downLoValues << thisMajorFreq + minorSize*direction*static_cast<double>(j);
        }
    }

    if(ftc)
    {
        ftc->d_downStart = start;
        ftc->d_downEnd = end;
        ftc->d_downMaj = numMajor;
        ftc->d_downMin = numMinor;
    }

    rfc.clearClockSteps();

    for(int i=0; i<upLoValues.size() && i<downLoValues.size(); i++)
        rfc.addLoScanClockStep(upLoValues.at(i),downLoValues.at(i));

    rfc.d_shotsPerClockConfig = p_shotsPerStepBox->value();
    rfc.d_targetSweeps = p_targetSweepsBox->value();
    
    return true;
}

bool WizardLoScanConfigPage::isComplete() const
{
    return true;
}

int WizardLoScanConfigPage::nextId() const
{
    return ExperimentWizard::ChirpConfigPage;
}

void WizardLoScanConfigPage::startChanged(RfConfig::ClockType t, double val)
{
    auto e = getExperiment();
    auto const &rfc = e->ftmwConfig()->d_rfConfig;

    if(rfc.d_commonUpDownLO && t == RfConfig::UpLO)
    {
        p_downStartBox->setValue(val);
    }

    QDoubleSpinBox *box = p_upMajorStepBox;
    if(t == RfConfig::DownLO)
        box = p_downMajorStepBox;

    box->blockSignals(true);
    box->setValue(calculateMajorStepSize(t));
    box->blockSignals(false);

}

void WizardLoScanConfigPage::endChanged(RfConfig::ClockType t, double val)
{
    auto e = getExperiment();
    auto const &rfc = e->ftmwConfig()->d_rfConfig;
    if(rfc.d_commonUpDownLO && t == RfConfig::UpLO)
    {
        p_downEndBox->setValue(val);
    }

    QDoubleSpinBox *box = p_upMajorStepBox;
    if(t == RfConfig::DownLO)
        box = p_downMajorStepBox;

    box->blockSignals(true);
    box->setValue(calculateMajorStepSize(t));
    box->blockSignals(false);

}

void WizardLoScanConfigPage::minorStepChanged(RfConfig::ClockType t, int val)
{
    Q_UNUSED(t)

    //number of steps must be equal between up and downconversion
    p_upNumMinorBox->blockSignals(true);
    p_upNumMinorBox->setValue(val);
    p_upNumMinorBox->blockSignals(false);

    p_downNumMinorBox->blockSignals(true);
    p_downNumMinorBox->setValue(val);
    p_downNumMinorBox->blockSignals(false);

    p_upMinorStepBox->setEnabled(val > 1);

    //calculate new end frequencies if necessary
    p_upEndBox->blockSignals(true);
    p_downEndBox->blockSignals(true);
    double start = p_upStartBox->value();
    double end = p_upEndBox->value();
    double step = p_upMinorStepBox->value();
    if(end < start)
        step*=-1.0;
    if(end + val*step > p_upEndBox->maximum())
        p_upEndBox->setValue(p_upEndBox->maximum());
    else if(end + val*step < p_upEndBox->minimum())
        p_upEndBox->setValue(p_upEndBox->minimum());

    p_downMinorStepBox->setEnabled(val > 1);
    start = p_downStartBox->value();
    end = p_downEndBox->value();
    step = p_downMinorStepBox->value();
    if(end < start)
        step*=-1.0;
    if(end + val*step > p_downEndBox->maximum())
        p_downEndBox->setValue(p_downEndBox->maximum());
    else if(end + val*step < p_downEndBox->minimum())
        p_upEndBox->setValue(p_downEndBox->minimum());
    p_upEndBox->blockSignals(false);
    p_downEndBox->blockSignals(false);

    //calculate new major step sizes
    QDoubleSpinBox *box = p_upMajorStepBox;
    if(t == RfConfig::DownLO)
        box = p_downMajorStepBox;

    box->blockSignals(true);
    box->setValue(calculateMajorStepSize(t));
    box->blockSignals(false);


}

void WizardLoScanConfigPage::minorStepSizeChanged(RfConfig::ClockType t, double val)
{
    auto e = getExperiment();
    auto const &rfc = e->ftmwConfig()->d_rfConfig;

    if(rfc.d_commonUpDownLO && t == RfConfig::UpLO)
    {
        p_downMinorStepBox->setValue(val);
    }

    p_upEndBox->blockSignals(true);
    p_downEndBox->blockSignals(true);
    if(t == RfConfig::UpLO)
    {
        double start = p_upStartBox->value();
        double end = p_upEndBox->value();
        int num = p_upNumMinorBox->value();
        if(end < start)
            val*=-1.0;
        if(end + num*val > p_upEndBox->maximum())
            p_upEndBox->setValue(p_upEndBox->maximum());
        else if(end + num*val < p_upEndBox->minimum())
            p_upEndBox->setValue(p_upEndBox->minimum());
    }
    else
    {
        double start = p_downStartBox->value();
        double end = p_downEndBox->value();
        int num = p_downNumMinorBox->value();
        if(end < start)
            val*=-1.0;
        if(end + num*val > p_downEndBox->maximum())
            p_downEndBox->setValue(p_downEndBox->maximum());
        else if(end + num*val < p_downEndBox->minimum())
            p_downEndBox->setValue(p_downEndBox->minimum());
    }
    p_upEndBox->blockSignals(false);
    p_downEndBox->blockSignals(false);

    //calculate new major step sizes
    QDoubleSpinBox *box = p_upMajorStepBox;
    if(t == RfConfig::DownLO)
        box = p_downMajorStepBox;

    box->blockSignals(true);
    box->setValue(calculateMajorStepSize(t));
    box->blockSignals(false);

}

void WizardLoScanConfigPage::majorStepChanged(RfConfig::ClockType t, int val)
{
    Q_UNUSED(t)

    //major steps must always be equal
    p_upNumMajorBox->blockSignals(true);
    p_upNumMajorBox->setValue(val);
    p_upNumMajorBox->blockSignals(false);

    p_downNumMajorBox->blockSignals(true);
    p_downNumMajorBox->setValue(val);
    p_downNumMajorBox->blockSignals(false);


    p_upMajorStepBox->blockSignals(true);
    p_upMajorStepBox->setValue(calculateMajorStepSize(RfConfig::UpLO));
    p_upMajorStepBox->blockSignals(false);

    p_downMajorStepBox->blockSignals(true);
    p_downMajorStepBox->setValue(calculateMajorStepSize(RfConfig::DownLO));
    p_downMajorStepBox->blockSignals(false);

}

void WizardLoScanConfigPage::majorStepSizeChanged(RfConfig::ClockType t, double val)
{
    double start = p_upStartBox->value();
    double end = p_upEndBox->value();
    double minorStep = p_upMinorStepBox->value();
    int numMinor = p_upNumMinorBox->value();
    int numMajor = p_upNumMajorBox->value();
    if(t == RfConfig::DownLO)
    {
        start = p_upStartBox->value();
        end = p_upEndBox->value();
        minorStep = p_upMinorStepBox->value();
        numMinor = p_upNumMinorBox->value();
        numMajor = p_upNumMajorBox->value();
    }

    //calculate number of major steps that fits with this step size
    if(end < start)
        minorStep*= -1.0;
    end -= static_cast<double>(numMinor-1)*minorStep;
    int newMajorStep = floor(fabs(end-start)/val)+1;

    if(newMajorStep != numMajor)
        p_upNumMajorBox->setValue(newMajorStep); //this will also update downNumMajor box

}

void WizardLoScanConfigPage::fixedChanged(bool fixed)
{
    p_downEndBox->setEnabled(!fixed);
    p_downMajorStepBox->setEnabled(!fixed);
    p_downMinorStepBox->setEnabled(!fixed);
    p_downNumMajorBox->setEnabled(!fixed);
    p_downNumMinorBox->setEnabled(!fixed);
    p_constantDownOffsetBox->setEnabled(!fixed);
}

void WizardLoScanConfigPage::constantOffsetChanged(bool co)
{
    p_downEndBox->setEnabled(!co);
    p_downMajorStepBox->setEnabled(!co);
    p_downMinorStepBox->setEnabled(!co);
    p_downNumMajorBox->setEnabled(!co);
    p_downNumMinorBox->setEnabled(!co);
    p_fixedDownLoBox->setEnabled(!co);
}

double WizardLoScanConfigPage::calculateMajorStepSize(RfConfig::ClockType t)
{
    if(t == RfConfig::UpLO)
    {
        //calculate new step size
        int numMinor = p_upNumMinorBox->value();
        double minorSize = p_upMinorStepBox->value();

        double start = p_upStartBox->value();
        double end = p_upEndBox->value();
        if(end < start)
            minorSize*=-1.0;

        double lastMajor = end - static_cast<double>(numMinor-1)*minorSize;
        int numMajor = p_upNumMajorBox->value();

        return fabs(lastMajor - start)/static_cast<double>(numMajor-1);


    }
    else
    {
        //calculate new step size
        int numMinor = p_downNumMinorBox->value();
        double minorSize = p_downMinorStepBox->value();

        double start = p_downStartBox->value();
        double end = p_downEndBox->value();

        if(end < start)
            minorSize*=-1.0;

        double lastMajor = end - static_cast<double>(numMinor-1)*minorSize;
        int numMajor = p_downNumMajorBox->value();

        return fabs(lastMajor - start)/static_cast<double>(numMajor-1);


    }
}
