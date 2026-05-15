#include "loscanconfigwidget.h"

#include <climits>

using namespace BC::Key::WizLoScan;

#include <QGroupBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QVBoxLayout>
#include <QFormLayout>
#include <QCheckBox>
#include <QTableWidget>

#include <data/experiment/experiment.h>
#include <data/experiment/ftmwconfigtypes.h>
namespace {
constexpr int UpCol = 0;
constexpr int DownCol = 1;
constexpr int StartRow = 0;
constexpr int EndRow = 1;
constexpr int MinorStepsRow = 2;
constexpr int MinorSizeRow = 3;
constexpr int MajorStepsRow = 4;
constexpr int MajorSizeRow = 5;
}

LOScanConfigWidget::LOScanConfigWidget(Experiment *exp, QWidget *parent)
    : QWidget(parent), SettingsStorage(BC::Key::WizLoScan::key), p_exp(exp)
{
    auto makeFreqBox = [](double defaultVal, double singleStep,
                          const QString &tip, double minRange = 0.0)
    {
        auto box = new QDoubleSpinBox;
        box->setDecimals(6);
        box->setSuffix(QString(" MHz"));
        box->setSingleStep(singleStep);
        box->setRange(minRange, 1e9);
        box->setValue(defaultVal);
        box->setToolTip(tip);
        box->setKeyboardTracking(false);
        box->setAlignment(Qt::AlignCenter);
        return box;
    };

    auto makeCountBox = [](int defaultVal, int minVal, int maxVal,
                           const QString &tip)
    {
        auto box = new QSpinBox;
        box->setRange(minVal, maxVal);
        box->setValue(defaultVal);
        box->setToolTip(tip);
        box->setAlignment(Qt::AlignCenter);
        return box;
    };

    p_upStartBox = makeFreqBox(get<double>(upStart, 0.0), 1000.0,
        QString("Starting major step LO frequency.\nChanging this value will update the major step size."));
    p_upEndBox = makeFreqBox(get<double>(upEnd, 1000.0), 1000.0,
        QString("Ending LO frequency (including major and minor steps).\nThis is a limit; the frequency will not exceed this value.\nChanging this value will update the major step size."));
    p_upMinorStepBox = makeFreqBox(get<double>(upMinorStep, 0.0), 1.0,
        QString("Minor step size, if number of minor steps > 1.\nMay change the number of major steps."));
    p_upMajorStepBox = makeFreqBox(get<double>(upMajorStep, 1000.0), 100.0,
        QString("Desired major step size.\nChanging this will update the number of major steps."));
    p_upNumMinorBox = makeCountBox(get<int>(upNumMinor, 1), 1, 10,
        QString("Number of minor (small) steps to take per major step.\nThe sign is determined automatically."));
    p_upNumMajorBox = makeCountBox(get<int>(upNumMajor, 2), 2, 100000,
        QString("Number of major steps desired.\nChanging this will update the major step size."));

    p_downStartBox = makeFreqBox(get<double>(downStart, 0.0), 1000.0,
        QString("Starting major step LO frequency."));
    p_downEndBox = makeFreqBox(get<double>(downEnd, 1000.0), 1000.0,
        QString("Ending LO frequency (including major and minor steps).\nThis is a limit; the frequency will not exceed this value.\nChanging this value will update the major step size."));
    p_downMinorStepBox = makeFreqBox(get<double>(downMinorStep, 0.0), 1.0,
        QString("Minor step size, if number of minor steps > 1.\nMay change the number of major steps."));
    p_downMajorStepBox = makeFreqBox(get<double>(downMajorStep, 1000.0), 100.0,
        QString("Desired major step size.\nChanging this will update the number of major steps."));
    p_downNumMinorBox = makeCountBox(get<int>(downNumMinor, 1), 1, 10,
        QString("Number of minor (small) steps to take per major step.\nThe sign is determined automatically."));
    p_downNumMajorBox = makeCountBox(get<int>(downNumMajor, 2), 2, 100000,
        QString("Number of major steps desired.\nChanging this will update the major step size."));

    p_loTable = new QTableWidget(6, 2, this);
    p_loTable->setHorizontalHeaderLabels({"Up LO", "Down LO"});
    p_loTable->setVerticalHeaderLabels({"Start", "End",
                                        "Minor Steps/pt", "Minor Step Size",
                                        "Major Steps", "Major Step Size"});
    p_loTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    p_loTable->verticalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    p_loTable->setSelectionMode(QAbstractItemView::NoSelection);
    p_loTable->setFocusPolicy(Qt::NoFocus);
    p_loTable->setShowGrid(true);

    p_loTable->setCellWidget(StartRow,      UpCol,   p_upStartBox);
    p_loTable->setCellWidget(EndRow,        UpCol,   p_upEndBox);
    p_loTable->setCellWidget(MinorStepsRow, UpCol,   p_upNumMinorBox);
    p_loTable->setCellWidget(MinorSizeRow,  UpCol,   p_upMinorStepBox);
    p_loTable->setCellWidget(MajorStepsRow, UpCol,   p_upNumMajorBox);
    p_loTable->setCellWidget(MajorSizeRow,  UpCol,   p_upMajorStepBox);

    p_loTable->setCellWidget(StartRow,      DownCol, p_downStartBox);
    p_loTable->setCellWidget(EndRow,        DownCol, p_downEndBox);
    p_loTable->setCellWidget(MinorStepsRow, DownCol, p_downNumMinorBox);
    p_loTable->setCellWidget(MinorSizeRow,  DownCol, p_downMinorStepBox);
    p_loTable->setCellWidget(MajorStepsRow, DownCol, p_downNumMajorBox);
    p_loTable->setCellWidget(MajorSizeRow,  DownCol, p_downMajorStepBox);

    p_fixedDownLoBox = new QCheckBox(QString("Fixed Frequency"));
    p_fixedDownLoBox->setToolTip(QString("If checked, the downconversion frequency will be set to the start value for all points."));
    p_fixedDownLoBox->setChecked(get<bool>(downFixed, false));

    p_constantDownOffsetBox = new QCheckBox(QString("Constant Offset"));
    p_constantDownOffsetBox->setToolTip(QString("If checked, the downconversion frequency will maintain a constant difference from the upconversion LO.\nThe difference will be kept at the difference of the start frequencies."));
    p_constantDownOffsetBox->setChecked(get<bool>(constOffset, false));

    auto *downModeRow = new QHBoxLayout;
    downModeRow->addWidget(new QLabel("Down LO Mode:"));
    downModeRow->addWidget(p_fixedDownLoBox);
    downModeRow->addWidget(p_constantDownOffsetBox);
    downModeRow->addStretch(1);

    auto *otherBox = new QGroupBox(QString("Scan Settings"));

    p_shotsPerStepBox = new QSpinBox;
    p_shotsPerStepBox->setRange(1,INT_MAX);
    p_shotsPerStepBox->setSingleStep(1000);
    p_shotsPerStepBox->setValue(get(shots,1000));
    p_shotsPerStepBox->setAlignment(Qt::AlignCenter);
    p_shotsPerStepBox->setToolTip(QString("Number of shots to acquire at each step (major and minor)."));

    p_targetSweepsBox = new QSpinBox;
    p_targetSweepsBox->setRange(1,INT_MAX);
    p_targetSweepsBox->setValue(get(sweeps,1));
    p_targetSweepsBox->setAlignment(Qt::AlignCenter);
    p_targetSweepsBox->setToolTip(QString("Number of sweeps through the total LO range.\nExperiment will end when this number is reached."));

    auto *settingsRow = new QHBoxLayout;
    auto *shotsLbl = new QLabel("Shots/Point:");
    shotsLbl->setAlignment(Qt::AlignRight|Qt::AlignVCenter);
    settingsRow->addWidget(shotsLbl);
    settingsRow->addWidget(p_shotsPerStepBox, 1);
    auto *sweepsLbl = new QLabel("Target Sweeps:");
    sweepsLbl->setAlignment(Qt::AlignRight|Qt::AlignVCenter);
    settingsRow->addWidget(sweepsLbl);
    settingsRow->addWidget(p_targetSweepsBox, 1);
    otherBox->setLayout(settingsRow);

    auto *vbl = new QVBoxLayout;
    vbl->addWidget(otherBox,0);
    vbl->addWidget(p_loTable,1);
    vbl->addLayout(downModeRow,0);

    setLayout(vbl);

    auto vc = static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged);
    auto dvc = static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged);

    connect(p_upStartBox,dvc,[this](double v){
        startChanged(RfConfig::UpLO,v);
    });
    connect(p_downStartBox,dvc,[this](double v){
        startChanged(RfConfig::DownLO,v);
    });

    connect(p_upEndBox,dvc,[this](double v){
        endChanged(RfConfig::UpLO,v);
    });
    connect(p_downEndBox,dvc,[this](double v){
        endChanged(RfConfig::DownLO,v);
    });

    connect(p_upMajorStepBox,dvc,[this](double v){
       majorStepSizeChanged(RfConfig::UpLO,v);
    });
    connect(p_downMajorStepBox,dvc,[this](double v){
       majorStepSizeChanged(RfConfig::DownLO,v);
    });

    connect(p_upMinorStepBox,dvc,[this](double v){
       minorStepSizeChanged(RfConfig::UpLO,v);
    });
    connect(p_downMinorStepBox,dvc,[this](double v){
       minorStepSizeChanged(RfConfig::DownLO,v);
    });

    connect(p_upNumMinorBox,vc,[this](int v){
       minorStepChanged(RfConfig::UpLO,v);
    });
    connect(p_downNumMinorBox,vc,[this](int v){
       minorStepChanged(RfConfig::DownLO,v);
    });

    connect(p_upNumMajorBox,vc,[this](int v){
       majorStepChanged(RfConfig::UpLO,v);
    });
    connect(p_downNumMajorBox,vc,[this](int v){
       majorStepChanged(RfConfig::DownLO,v);
    });

    connect(p_constantDownOffsetBox,&QCheckBox::toggled,this,&LOScanConfigWidget::constantOffsetChanged);
    connect(p_fixedDownLoBox,&QCheckBox::toggled,this,&LOScanConfigWidget::fixedChanged);

    registerGetter(shots,p_shotsPerStepBox,&QSpinBox::value);
    registerGetter(sweeps,p_targetSweepsBox,&QSpinBox::value);

    registerGetter(upStart,p_upStartBox,&QDoubleSpinBox::value);
    registerGetter(upEnd,p_upEndBox,&QDoubleSpinBox::value);
    registerGetter(upNumMinor,p_upNumMinorBox,&QSpinBox::value);
    registerGetter(upMinorStep,p_upMinorStepBox,&QDoubleSpinBox::value);
    registerGetter(upNumMajor,p_upNumMajorBox,&QSpinBox::value);
    registerGetter(upMajorStep,p_upMajorStepBox,&QDoubleSpinBox::value);

    registerGetter(downStart,p_downStartBox,&QDoubleSpinBox::value);
    registerGetter(downEnd,p_downEndBox,&QDoubleSpinBox::value);
    registerGetter(downNumMinor,p_downNumMinorBox,&QSpinBox::value);
    registerGetter(downMinorStep,p_downMinorStepBox,&QDoubleSpinBox::value);
    registerGetter(downNumMajor,p_downNumMajorBox,&QSpinBox::value);
    registerGetter(downMajorStep,p_downMajorStepBox,&QDoubleSpinBox::value);

    registerGetter(downFixed,static_cast<QAbstractButton*>(p_fixedDownLoBox),&QCheckBox::isChecked);
    registerGetter(constOffset,static_cast<QAbstractButton*>(p_constantDownOffsetBox),&QCheckBox::isChecked);

    if(p_exp->d_number > 0)
    {
        initialize();
        auto const &rfc = p_exp->ftmwConfig()->d_rfConfig;
        p_targetSweepsBox->setValue(rfc.d_targetSweeps);
        p_shotsPerStepBox->setValue(rfc.d_shotsPerClockConfig);
        auto ftc = dynamic_cast<FtmwConfigLOScan*>(p_exp->ftmwConfig());
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
}

void LOScanConfigWidget::startChanged(RfConfig::ClockType t, double val)
{
    if(!p_exp->ftmwConfig())
        return;

    auto const &rfc = p_exp->ftmwConfig()->d_rfConfig;

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

void LOScanConfigWidget::endChanged(RfConfig::ClockType t, double val)
{
    if(!p_exp->ftmwConfig())
        return;

    auto const &rfc = p_exp->ftmwConfig()->d_rfConfig;
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

void LOScanConfigWidget::minorStepChanged(RfConfig::ClockType t, int val)
{
    Q_UNUSED(t)

    p_upNumMinorBox->blockSignals(true);
    p_upNumMinorBox->setValue(val);
    p_upNumMinorBox->blockSignals(false);

    p_downNumMinorBox->blockSignals(true);
    p_downNumMinorBox->setValue(val);
    p_downNumMinorBox->blockSignals(false);

    p_upMinorStepBox->setEnabled(val > 1);

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

    QDoubleSpinBox *box = p_upMajorStepBox;
    if(t == RfConfig::DownLO)
        box = p_downMajorStepBox;

    box->blockSignals(true);
    box->setValue(calculateMajorStepSize(t));
    box->blockSignals(false);
}

void LOScanConfigWidget::minorStepSizeChanged(RfConfig::ClockType t, double val)
{
    if(!p_exp->ftmwConfig())
        return;

    auto const &rfc = p_exp->ftmwConfig()->d_rfConfig;

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

    QDoubleSpinBox *box = p_upMajorStepBox;
    if(t == RfConfig::DownLO)
        box = p_downMajorStepBox;

    box->blockSignals(true);
    box->setValue(calculateMajorStepSize(t));
    box->blockSignals(false);
}

void LOScanConfigWidget::majorStepChanged(RfConfig::ClockType t, int val)
{
    Q_UNUSED(t)

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

void LOScanConfigWidget::majorStepSizeChanged(RfConfig::ClockType t, double val)
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

    if(end < start)
        minorStep*= -1.0;
    end -= static_cast<double>(numMinor-1)*minorStep;
    int newMajorStep = floor(fabs(end-start)/val)+1;

    if(newMajorStep != numMajor)
        p_upNumMajorBox->setValue(newMajorStep);
}

void LOScanConfigWidget::fixedChanged(bool fixed)
{
    p_downEndBox->setEnabled(!fixed);
    p_downMajorStepBox->setEnabled(!fixed);
    p_downMinorStepBox->setEnabled(!fixed);
    p_downNumMajorBox->setEnabled(!fixed);
    p_downNumMinorBox->setEnabled(!fixed);
    p_constantDownOffsetBox->setEnabled(!fixed);
}

void LOScanConfigWidget::constantOffsetChanged(bool co)
{
    p_downEndBox->setEnabled(!co);
    p_downMajorStepBox->setEnabled(!co);
    p_downMinorStepBox->setEnabled(!co);
    p_downNumMajorBox->setEnabled(!co);
    p_downNumMinorBox->setEnabled(!co);
    p_fixedDownLoBox->setEnabled(!co);
}

double LOScanConfigWidget::calculateMajorStepSize(RfConfig::ClockType t)
{
    if(t == RfConfig::UpLO)
    {
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

LOScanConfigWidget::LoRanges LOScanConfigWidget::calculateLoRanges() const
{
    if(!p_exp->ftmwConfig())
        return {{0.0,1.0e7},{0.0,1.0e7}};
    auto const &rfc = p_exp->ftmwConfig()->d_rfConfig;

    return {{rfc.clockRange(RfConfig::UpLO)},{rfc.clockRange(RfConfig::DownLO)}};
}

void LOScanConfigWidget::initialize()
{
    if(!isEnabled() || !p_exp->ftmwConfig())
        return;

    auto const &rfc = p_exp->ftmwConfig()->d_rfConfig;

    auto s = calculateLoRanges();
    auto upMinFreq = s.upLoRange.first;
    auto upMaxFreq = s.upLoRange.second;
    auto downMinFreq = s.downLoRange.first;
    auto downMaxFreq = s.downLoRange.second;

    p_upStartBox->setRange(upMinFreq,upMaxFreq);
    p_upEndBox->setRange(upMinFreq,upMaxFreq);
    p_upMajorStepBox->setRange(1.0,upMaxFreq-upMinFreq);
    p_upMinorStepBox->setRange(0.0,upMaxFreq-upMinFreq);

    p_downStartBox->setRange(downMinFreq,downMaxFreq);
    p_downEndBox->setRange(downMinFreq,downMaxFreq);
    p_downMajorStepBox->setRange(1.0,downMaxFreq-downMinFreq);
    p_downMinorStepBox->setRange(0.0,downMaxFreq-downMinFreq);

    setDownColumnEnabled(!rfc.d_commonUpDownLO);
    p_fixedDownLoBox->setEnabled(!rfc.d_commonUpDownLO);
    p_constantDownOffsetBox->setEnabled(!rfc.d_commonUpDownLO);

    if(!rfc.d_commonUpDownLO)
    {
        if(p_fixedDownLoBox->isChecked())
            fixedChanged(true);
        else if(p_constantDownOffsetBox->isChecked())
            constantOffsetChanged(true);
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

bool LOScanConfigWidget::validate()
{
    if(!isEnabled() || !p_exp->ftmwConfig())
        return true;

    auto s = calculateLoRanges();
    bool out = true;

    if(p_upStartBox->value() < s.upLoRange.first || p_upStartBox->value() > s.upLoRange.second)
    {
        emit error("Starting LO Frequency is out of range of upconversion LO.");
        out = false;
    }
    if(p_upEndBox->value() < s.upLoRange.first || p_upEndBox->value() > s.upLoRange.second)
    {
        emit error("Ending LO Frequency is out of range of upconversion LO.");
        out = false;
    }
    if(p_downStartBox->value() < s.downLoRange.first || p_downStartBox->value() > s.downLoRange.second)
    {
        emit error("Starting LO Frequency is out of range of downconversion LO.");
        out = false;
    }
    if(p_downEndBox->value() < s.downLoRange.first || p_downEndBox->value() > s.downLoRange.second)
    {
        emit error("Ending LO Frequency is out of range of downconversion LO.");
        out = false;
    }

    return out;
}

void LOScanConfigWidget::apply()
{
    if(!isEnabled() || !p_exp->ftmwConfig())
        return;

    auto &rfc = p_exp->ftmwConfig()->d_rfConfig;
    auto ftc = dynamic_cast<FtmwConfigLOScan*>(p_exp->ftmwConfig());

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
}

void LOScanConfigWidget::setDownColumnEnabled(bool enabled)
{
    for(int row = 0; row < p_loTable->rowCount(); ++row)
    {
        if(auto w = p_loTable->cellWidget(row, DownCol))
            w->setEnabled(enabled);
    }
}
