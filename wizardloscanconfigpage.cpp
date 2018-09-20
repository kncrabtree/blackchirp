#include "wizardloscanconfigpage.h"

#include <QGroupBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QGridLayout>
#include <QSettings>
#include <QLabel>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QCheckBox>

WizardLoScanConfigPage::WizardLoScanConfigPage(QWidget *parent) : ExperimentWizardPage(parent)
{
    setTitle(QString("Configure LO Scan"));
    setSubTitle(QString("Hover over the various fields for more information."));

    p_upBox = new QGroupBox("Upconversion LO");

    p_upStartBox = new QDoubleSpinBox;
    p_upStartBox->setDecimals(6);
    p_upStartBox->setSuffix(QString(" MHz"));
    p_upStartBox->setSingleStep(1000.0);
    p_upStartBox->setRange(0.0,1e9);
    p_upStartBox->setToolTip(QString("Starting major step LO frequency.\nChanging this value will update the major step size."));
    p_upStartBox->setKeyboardTracking(false);

    p_upEndBox = new QDoubleSpinBox;
    p_upEndBox->setDecimals(6);
    p_upEndBox->setSuffix(QString(" MHz"));
    p_upEndBox->setSingleStep(1000.0);
    p_upEndBox->setRange(0.0,1e9);
    p_upEndBox->setToolTip(QString("Ending LO frequency (including major and minor steps).\nThis is a limit; the frequency will not exceed this value.\nChanging this value will update the major step size."));
    p_upEndBox->setKeyboardTracking(false);

    p_upNumMinorBox = new QSpinBox;
    p_upNumMinorBox->setRange(1,10);
    p_upNumMinorBox->setToolTip(QString("Number of minor (small) steps to take per major step.\nThe sign is determined automatically."));

    p_upMinorStepBox = new QDoubleSpinBox;
    p_upMinorStepBox->setDecimals(6);
    p_upMinorStepBox->setSuffix(QString(" MHz"));
    p_upMinorStepBox->setSingleStep(1.0);
    p_upMinorStepBox->setRange(0.0,1e9);
    p_upMinorStepBox->setToolTip(QString("Minor step size, if number of minor steps > 1.\nMay change the number of major steps."));
    p_upMinorStepBox->setKeyboardTracking(false);

    p_upNumMajorBox = new QSpinBox;
    p_upNumMajorBox->setRange(2,100000);
    p_upNumMajorBox->setToolTip(QString("Number of major steps desired.\nChanging this will update the major step size."));

    p_upMajorStepBox = new QDoubleSpinBox;
    p_upMajorStepBox->setDecimals(6);
    p_upMajorStepBox->setSuffix(QString(" MHz"));
    p_upMajorStepBox->setSingleStep(100.0);
    p_upMajorStepBox->setRange(0.0,1e9);
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

    p_upBox->setLayout(upgl);



    p_downBox = new QGroupBox("Upconversion LO");

    p_downStartBox = new QDoubleSpinBox;
    p_downStartBox->setDecimals(6);
    p_downStartBox->setSuffix(QString(" MHz"));
    p_downStartBox->setSingleStep(1000.0);
    p_downStartBox->setRange(0.0,1e9);
    p_downStartBox->setToolTip(QString("Starting major step LO frequency."));
    p_downStartBox->setKeyboardTracking(false);

    p_downEndBox = new QDoubleSpinBox;
    p_downEndBox->setDecimals(6);
    p_downEndBox->setSuffix(QString(" MHz"));
    p_downEndBox->setSingleStep(1000.0);
    p_downEndBox->setRange(0.0,1e9);
    p_downEndBox->setToolTip(QString("Ending LO frequency (including major and minor steps).\nThis is a limit; the frequency will not exceed this value.\nChanging this value will update the major step size."));
    p_downEndBox->setKeyboardTracking(false);

    p_downNumMinorBox = new QSpinBox;
    p_downNumMinorBox->setRange(1,10);
    p_downNumMinorBox->setToolTip(QString("Number of minor (small) steps to take per major step.\nThe sign is determined automatically."));

    p_downMinorStepBox = new QDoubleSpinBox;
    p_downMinorStepBox->setDecimals(6);
    p_downMinorStepBox->setSuffix(QString(" MHz"));
    p_downMinorStepBox->setSingleStep(1.0);
    p_downMinorStepBox->setRange(0.0,1e9);
    p_downMinorStepBox->setToolTip(QString("Minor step size, if number of minor steps > 1.\nMay change the number of major steps."));
    p_downMinorStepBox->setKeyboardTracking(false);

    p_downNumMajorBox = new QSpinBox;
    p_downNumMajorBox->setRange(2,100000);
    p_downNumMajorBox->setToolTip(QString("Number of major steps desired.\nChanging this will update the major step size."));

    p_downMajorStepBox = new QDoubleSpinBox;
    p_downMajorStepBox->setDecimals(6);
    p_downMajorStepBox->setSuffix(QString(" MHz"));
    p_downMajorStepBox->setSingleStep(100.0);
    p_downMajorStepBox->setRange(0.0,1e9);
    p_downMajorStepBox->setToolTip(QString("Desired major step size.\nChanging this will update the number of major steps."));
    p_downMajorStepBox->setKeyboardTracking(false);

    p_fixedDownLoBox = new QCheckBox(QString("Fixed Frequency"));
    p_fixedDownLoBox->setToolTip(QString("If checked, the downconversion frequency will be set to the start value for all points."));

    p_constantDownOffsetBox = new QCheckBox(QString("Constant Offset"));
    p_constantDownOffsetBox->setToolTip(QString("If checked, the downconversion frequency will maintain a constant difference from the upconversion LO.\nThe difference will be kept at the difference of the start frequencies."));

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

    p_downBox->setLayout(downgl);

    auto *otherBox = new QGroupBox(QString("Scan Settings"));
    auto *fl = new QFormLayout;

    p_shotsPerStepBox = new QSpinBox;
    p_shotsPerStepBox->setRange(1,__INT_MAX__);
    p_shotsPerStepBox->setSingleStep(1000);
    p_shotsPerStepBox->setToolTip(QString("Number of shots to acquire at each step (major and minor)."));
    fl->addRow(QString("Shots/Point"),p_shotsPerStepBox);


    p_targetSweepsBox = new QSpinBox;
    p_targetSweepsBox->setRange(1,__INT_MAX__);
    p_targetSweepsBox->setToolTip(QString("Number of sweeps through the total LO range.\nExperiment will end when this number is reached."));
    fl->addRow(QString("Target Sweeps"),p_targetSweepsBox);
    otherBox->setLayout(fl);

    auto *hbl = new QHBoxLayout;
    hbl->addWidget(otherBox);
    hbl->addWidget(p_upBox);
    hbl->addWidget(p_downBox);

    setLayout(hbl);

    auto vc = static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged);
    auto dvc = static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged);

    connect(p_upStartBox,dvc,[=](double v){
        startChanged(BlackChirp::UpConversionLO,v);
    });
    connect(p_downStartBox,dvc,[=](double v){
        startChanged(BlackChirp::DownConversionLO,v);
    });

    connect(p_upEndBox,dvc,[=](double v){
        endChanged(BlackChirp::UpConversionLO,v);
    });
    connect(p_downEndBox,dvc,[=](double v){
        endChanged(BlackChirp::DownConversionLO,v);
    });

    connect(p_upMajorStepBox,dvc,[=](double v){
       majorStepSizeChanged(BlackChirp::UpConversionLO,v);
    });
    connect(p_downMajorStepBox,dvc,[=](double v){
       majorStepSizeChanged(BlackChirp::DownConversionLO,v);
    });

    connect(p_upMinorStepBox,dvc,[=](double v){
       minorStepSizeChanged(BlackChirp::UpConversionLO,v);
    });
    connect(p_downMinorStepBox,dvc,[=](double v){
       minorStepSizeChanged(BlackChirp::DownConversionLO,v);
    });

    connect(p_upNumMinorBox,vc,[=](int v){
       minorStepChanged(BlackChirp::UpConversionLO,v);
    });
    connect(p_downNumMinorBox,vc,[=](int v){
       minorStepChanged(BlackChirp::DownConversionLO,v);
    });

    connect(p_upNumMajorBox,vc,[=](int v){
       majorStepChanged(BlackChirp::UpConversionLO,v);
    });
    connect(p_downNumMajorBox,vc,[=](int v){
       majorStepChanged(BlackChirp::DownConversionLO,v);
    });

    connect(p_constantDownOffsetBox,&QCheckBox::toggled,this,&WizardLoScanConfigPage::constantOffsetChanged);
    connect(p_fixedDownLoBox,&QCheckBox::toggled,this,&WizardLoScanConfigPage::fixedChanged);

    loadFromSettings();

}

void WizardLoScanConfigPage::initializePage()
{
    auto e = getExperiment();
    d_rfConfig = e.ftmwConfig().rfConfig();

    //get LO hardware
    auto upLO = d_rfConfig.clockHardware(BlackChirp::UpConversionLO);
    auto downLO = d_rfConfig.clockHardware(BlackChirp::DownConversionLO);

    if(upLO.isEmpty())
        return;

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(upLO);
    s.beginGroup(s.value(QString("subKey"),QString("fixed")).toString());

    double upMinFreq = s.value(QString("minFreqMHz"),0.0).toDouble();
    double upMaxFreq = s.value(QString("maxFreqMHz"),1e7).toDouble();

    s.endGroup();
    s.endGroup();

    double downMinFreq = upMinFreq;
    double downMaxFreq = upMaxFreq;

    if(!d_rfConfig.commonLO() && upLO != downLO)
    {
        s.beginGroup(downLO);
        s.beginGroup(s.value(QString("subKey"),QString("fixed")).toString());
        downMinFreq = s.value(QString("minFreqMHz"),0.0).toDouble();
        downMaxFreq = s.value(QString("maxFreqMHz"),1e7).toDouble();

        s.endGroup();
        s.endGroup();
    }

    p_upStartBox->setRange(upMinFreq,upMaxFreq);
    p_upEndBox->setRange(upMinFreq,upMaxFreq);
    p_upMajorStepBox->setRange(1.0,upMaxFreq-upMinFreq);
    p_upMinorStepBox->setRange(0.0,upMaxFreq-upMinFreq);

    p_downStartBox->setRange(downMinFreq,downMaxFreq);
    p_downEndBox->setRange(downMinFreq,downMaxFreq);
    p_downMajorStepBox->setRange(1.0,downMaxFreq-downMinFreq);
    p_downMinorStepBox->setRange(0.0,downMaxFreq-downMinFreq);

    p_shotsPerStepBox->setValue(d_rfConfig.shotsPerClockStep());
    p_targetSweepsBox->setValue(d_rfConfig.targetSweeps());

    p_downBox->setDisabled(d_rfConfig.commonLO());
    if(d_rfConfig.commonLO())
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

    p_targetSweepsBox->setValue(d_rfConfig.targetSweeps());
    p_shotsPerStepBox->setValue(d_rfConfig.shotsPerClockStep());
}

bool WizardLoScanConfigPage::validatePage()
{
    auto e = getExperiment();

    QList<double> upLoValues, downLoValues;
    double direction = 1.0;
    double start = p_upStartBox->value();
    double end = p_upEndBox->value();
    int numMinor = p_upNumMinorBox->value();
    double minorSize = p_upMinorStepBox->value();
    double majorStep = p_upMajorStepBox->value();

    if(end < start)
        direction *= -1.0;

    for(int i=0; i<p_upNumMajorBox->value(); i++)
    {
        double thisMajorFreq = start + direction*majorStep*static_cast<double>(i);
        upLoValues << thisMajorFreq;
        for(int j=1; j<numMinor; j++)
            upLoValues << thisMajorFreq + minorSize*direction*static_cast<double>(j);
    }

    double offset = p_downStartBox->value() - start;
    direction = 1.0;
    start = p_downStartBox->value();
    end = p_downEndBox->value();
    numMinor = p_downNumMinorBox->value();
    minorSize = p_downMinorStepBox->value();
    majorStep = p_downMajorStepBox->value();
    if(end < start)
        direction *= -1.0;

    if(d_rfConfig.commonLO())
        downLoValues = upLoValues;
    else if(p_fixedDownLoBox->isChecked())
    {
        for(int i=0; i<upLoValues.size(); i++)
            downLoValues << start;
    }
    else if(p_constantDownOffsetBox->isChecked())
    {
        for(int i=0; i<upLoValues.size(); i++)
            downLoValues << upLoValues.at(i) + offset;
    }
    else
    {
        for(int i=0; i<p_downNumMajorBox->value(); i++)
        {
            double thisMajorFreq = start + direction*majorStep*static_cast<double>(i);
            upLoValues << thisMajorFreq;
            for(int j=1; j<numMinor; j++)
                upLoValues << thisMajorFreq + minorSize*direction*static_cast<double>(j);
        }
    }

    d_rfConfig.clearClockSteps();

    for(int i=0; i<upLoValues.size() && i<downLoValues.size(); i++)
        d_rfConfig.addClockStep(upLoValues.at(i),downLoValues.at(i));

    d_rfConfig.setShotsPerClockStep(p_shotsPerStepBox->value());
    d_rfConfig.setTargetSweeps(p_targetSweepsBox->value());

    e.setRfConfig(d_rfConfig);
    emit experimentUpdate(e);

    saveToSettings();
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

void WizardLoScanConfigPage::startChanged(BlackChirp::ClockType t, double val)
{
    if(d_rfConfig.commonLO() && t == BlackChirp::UpConversionLO)
    {
        p_downStartBox->setValue(val);
    }

    QDoubleSpinBox *box = p_upMajorStepBox;
    if(t == BlackChirp::DownConversionLO)
        box = p_downMajorStepBox;

    box->blockSignals(true);
    box->setValue(calculateMajorStepSize(t));
    box->blockSignals(false);

}

void WizardLoScanConfigPage::endChanged(BlackChirp::ClockType t, double val)
{
    if(d_rfConfig.commonLO() && t == BlackChirp::UpConversionLO)
    {
        p_downEndBox->setValue(val);
    }

    QDoubleSpinBox *box = p_upMajorStepBox;
    if(t == BlackChirp::DownConversionLO)
        box = p_downMajorStepBox;

    box->blockSignals(true);
    box->setValue(calculateMajorStepSize(t));
    box->blockSignals(false);

}

void WizardLoScanConfigPage::minorStepChanged(BlackChirp::ClockType t, int val)
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
    if(t == BlackChirp::DownConversionLO)
        box = p_downMajorStepBox;

    box->blockSignals(true);
    box->setValue(calculateMajorStepSize(t));
    box->blockSignals(false);


}

void WizardLoScanConfigPage::minorStepSizeChanged(BlackChirp::ClockType t, double val)
{
    if(d_rfConfig.commonLO() && t == BlackChirp::UpConversionLO)
    {
        p_downMinorStepBox->setValue(val);
    }

    p_upEndBox->blockSignals(true);
    p_downEndBox->blockSignals(true);
    if(t == BlackChirp::UpConversionLO)
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
    if(t == BlackChirp::DownConversionLO)
        box = p_downMajorStepBox;

    box->blockSignals(true);
    box->setValue(calculateMajorStepSize(t));
    box->blockSignals(false);

}

void WizardLoScanConfigPage::majorStepChanged(BlackChirp::ClockType t, int val)
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
    p_upMajorStepBox->setValue(calculateMajorStepSize(BlackChirp::UpConversionLO));
    p_upMajorStepBox->blockSignals(false);

    p_downMajorStepBox->blockSignals(true);
    p_downMajorStepBox->setValue(calculateMajorStepSize(BlackChirp::DownConversionLO));
    p_downMajorStepBox->blockSignals(false);

}

void WizardLoScanConfigPage::majorStepSizeChanged(BlackChirp::ClockType t, double val)
{
    double start = p_upStartBox->value();
    double end = p_upEndBox->value();
    double minorStep = p_upMinorStepBox->value();
    int numMinor = p_upNumMinorBox->value();
    int numMajor = p_upNumMajorBox->value();
    if(t == BlackChirp::DownConversionLO)
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

void WizardLoScanConfigPage::saveToSettings() const
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(QString("lastLoScan"));

    s.setValue(QString("upStart"),p_upStartBox->value());
    s.setValue(QString("upEnd"),p_upEndBox->value());
    s.setValue(QString("upNumMinor"),p_upNumMinorBox->value());
    s.setValue(QString("upMinorStep"),p_upMinorStepBox->value());
    s.setValue(QString("upNumMajor"),p_upNumMajorBox->value());
    s.setValue(QString("upMajorStep"),p_upMajorStepBox->value());

    s.setValue(QString("downStart"),p_downStartBox->value());
    s.setValue(QString("downEnd"),p_downEndBox->value());
    s.setValue(QString("downNumMinor"),p_downNumMinorBox->value());
    s.setValue(QString("downMinorStep"),p_downMinorStepBox->value());
    s.setValue(QString("downNumMajor"),p_downNumMajorBox->value());
    s.setValue(QString("downMajorStep"),p_downMajorStepBox->value());

    s.setValue(QString("downFixed"),p_fixedDownLoBox->isChecked());
    s.setValue(QString("downConstantOffset"),p_constantDownOffsetBox->isChecked());

    s.endGroup();
    s.sync();
}

void WizardLoScanConfigPage::loadFromSettings()
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(QString("lastLoScan"));

    p_upStartBox->setValue(s.value(QString("upStart"),0).toDouble());
    p_upEndBox->setValue(s.value(QString("upEnd"),1000.0).toDouble());
    p_upNumMinorBox->setValue(s.value(QString("upNumMinor"),1).toInt());
    p_upMinorStepBox->setValue(s.value(QString("upMinorStep"),0.0).toDouble());
    p_upNumMajorBox->setValue(s.value(QString("upNumMajor"),2).toInt());
    p_upMajorStepBox->setValue(s.value(QString("upMajorStep"),1000.0).toDouble());

    p_downStartBox->setValue(s.value(QString("downStart"),0).toDouble());
    p_downEndBox->setValue(s.value(QString("downEnd"),1000.0).toDouble());
    p_downNumMinorBox->setValue(s.value(QString("downNumMinor"),1).toInt());
    p_downMinorStepBox->setValue(s.value(QString("downMinorStep"),0.0).toDouble());
    p_downNumMajorBox->setValue(s.value(QString("downNumMajor"),2).toInt());
    p_downMajorStepBox->setValue(s.value(QString("downMajorStep"),1000.0).toDouble());

    p_fixedDownLoBox->setChecked(s.value(QString("downFixed"),false).toBool());
    p_constantDownOffsetBox->setChecked(s.value(QString("downConstantOffset"),false).toBool());

    s.endGroup();
}

double WizardLoScanConfigPage::calculateMajorStepSize(BlackChirp::ClockType t)
{
    if(t == BlackChirp::UpConversionLO)
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
