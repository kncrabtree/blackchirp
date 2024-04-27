#include "experimenttypepage.h"

#include <data/experiment/experiment.h>

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QFormLayout>
#include <QGridLayout>
#include <QCheckBox>
#include <QSpinBox>
#include <QGroupBox>
#include <QDoubleSpinBox>
#include <QComboBox>
#include <QLabel>
#include <QMetaEnum>
#include <QMessageBox>

#include <hardware/core/ftmwdigitizer/ftmwscope.h>

#ifdef BC_LIF
#include <hardware/core/hardwaremanager.h>
#include <hardware/optional/pulsegenerator/pulsegenerator.h>
#include <modules/lif/hardware/liflaser/liflaser.h>
#endif

using namespace BC::Key::WizStart;

ExperimentTypePage::ExperimentTypePage(Experiment *exp, QWidget *parent) :
    ExperimentConfigPage(key,title,exp,parent)
{
    QFormLayout *fl = new QFormLayout(this);

    p_ftmw = new QGroupBox(QString("FTMW"),this);
    p_ftmw->setCheckable(true);
    p_ftmw->setChecked(get(ftmw,true));
    registerGetter(ftmw,p_ftmw,&QGroupBox::isChecked);
    connect(p_ftmw,&QGroupBox::toggled,this,&ExperimentTypePage::typeChanged);

    p_ftmwTypeBox = new QComboBox(this);
    auto t = QMetaEnum::fromType<FtmwConfig::FtmwType>();
    auto num = t.keyCount();
    for(int i=0; i<num; ++i)
        p_ftmwTypeBox->addItem(QString(t.key(i)).replace(QChar('_'),QChar(' ')),
                               static_cast<FtmwConfig::FtmwType>(t.value(i)));

    p_ftmwTypeBox->setCurrentIndex(get(ftmwType,0));
    registerGetter(ftmwType,p_ftmwTypeBox,&QComboBox::currentIndex);
    connect(p_ftmwTypeBox,qOverload<int>(&QComboBox::currentIndexChanged),this,&ExperimentTypePage::typeChanged);
    fl->addRow("Type",p_ftmwTypeBox);

    p_ftmwShotsBox = new QSpinBox(this);
    p_ftmwShotsBox->setRange(1,__INT_MAX__);
    p_ftmwShotsBox->setToolTip(QString("Number of FIDs to average.\n"
                                       "When this number is reached, the experiment ends in Target Shots mode, while an exponentially weighted moving average engages in Peak Up mode.\n\n"
                                       "If this box is disabled, it is either irrelevant or will be configured on a later page (e.g. in LO/DR Scan mode)."));
    p_ftmwShotsBox->setSingleStep(5000);
    p_ftmwShotsBox->setValue(get(ftmwShots,10000));
    registerGetter(ftmwShots,p_ftmwShotsBox,&QSpinBox::value);
    fl->addRow("Shots",p_ftmwShotsBox);

    p_ftmwTargetDurationBox = new QSpinBox(this);
    p_ftmwTargetDurationBox->setRange(1,__INT_MAX__);
    p_ftmwTargetDurationBox->setSingleStep(30);
    p_ftmwTargetDurationBox->setSuffix(" min");
    p_ftmwTargetDurationBox->setToolTip(QString("Duration of experiment in Target Duration mode."));
    p_ftmwTargetDurationBox->setValue(get(ftmwDuration,60));
    registerGetter(ftmwDuration,p_ftmwTargetDurationBox,&QSpinBox::value);
    fl->addRow("Duration",p_ftmwTargetDurationBox);

    p_endTimeLabel = new QLabel;
    p_endTimeLabel->setAlignment(Qt::AlignLeft|Qt::AlignVCenter);
    // p_endTimeLabel->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Expanding);
    auto dt = QDateTime::currentDateTime().addSecs(p_ftmwTargetDurationBox->value()*60);
    p_endTimeLabel->setText(d_endText.arg(dt.toString("yyyy-MM-dd h:mm AP")));
    fl->addRow(p_endTimeLabel);
    connect(p_ftmwTargetDurationBox,qOverload<int>(&QSpinBox::valueChanged),this,&ExperimentTypePage::updateLabel);

    p_phaseCorrectionBox = new QCheckBox(this);
    p_phaseCorrectionBox->setToolTip(QString("If checked, Blackchirp will optimize the autocorrelation of the chirp during the acquisition.\n\nFor this to work, the chirp must be part of the signal recorded by the digitizer and must not saturate the digitizer."));
    p_phaseCorrectionBox->setChecked(get(ftmwPhase,false));
    registerGetter<QAbstractButton>(ftmwPhase,p_phaseCorrectionBox,
                   &QCheckBox::isChecked);
    fl->addRow("Phase Correction",p_phaseCorrectionBox);

    p_chirpScoringBox = new QCheckBox(this);
    p_chirpScoringBox->setToolTip(QString("If checked, Blackchirp will compare the RMS of the chirp in each new waveform with that of the current average chirp RMS.\nIf less than threshold*averageRMS, the FID will be rejected.\n\nFor this to work, the chirp must be part of the signal recorded by the digitizer and must not saturate the digitizer."));
    p_chirpScoringBox->setChecked(get(ftmwScoring,false));
    registerGetter<QAbstractButton>(ftmwScoring,p_chirpScoringBox,&QCheckBox::isChecked);
    fl->addRow("Chirp Scoring",p_chirpScoringBox);

    p_thresholdBox = new QDoubleSpinBox(this);
    p_thresholdBox->setRange(0.0,1.0);
    p_thresholdBox->setSingleStep(0.05);
    p_thresholdBox->setDecimals(3);
    p_thresholdBox->setValue(get(ftmwThresh,0.9));
    registerGetter(ftmwThresh,p_thresholdBox,&QDoubleSpinBox::value);
    fl->addRow("Chirp Threshold",p_thresholdBox);

    p_chirpOffsetBox = new QDoubleSpinBox(this);
    p_chirpOffsetBox->setRange(-0.00001,100.0);
    p_chirpOffsetBox->setDecimals(5);
    p_chirpOffsetBox->setSingleStep(0.1);
    p_chirpOffsetBox->setSuffix(QString::fromUtf16(u" μs"));
    p_chirpOffsetBox->setSpecialValueText(QString("Automatic"));
    p_chirpOffsetBox->setToolTip(QString("The time at which the chirp starts (used for phase correction and chirp scoring).\n\nIf automatic, Blackchirp assumes the digitizer is triggered at the start of the protection pulse,\nand accounts for the digitizer trigger position."));
    p_chirpOffsetBox->setValue(get(ftmwOffset,p_chirpOffsetBox->minimum()));
    registerGetter(ftmwOffset,p_chirpOffsetBox,&QDoubleSpinBox::value);
    fl->addRow("Chirp Start",p_chirpOffsetBox);

    for(int i=0; i<fl->rowCount(); ++i)
    {
        auto w = fl->itemAt(i,QFormLayout::LabelRole);
        if(w != nullptr)
        {
            auto lbl = dynamic_cast<QLabel*>(w->widget());
            if(lbl != nullptr)
            {
                lbl->setAlignment(Qt::AlignRight|Qt::AlignCenter);
                // lbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Expanding);
            }
        }
    }
    p_ftmw->setLayout(fl);

    auto *fl2 = new QFormLayout(this);

    auto *sgb = new QGroupBox(QString("Common Settings"));
    p_auxDataIntervalBox = new QSpinBox(this);
    p_auxDataIntervalBox->setRange(5,__INT_MAX__);
    p_auxDataIntervalBox->setSingleStep(300);
    p_auxDataIntervalBox->setSuffix(QString(" s"));
    p_auxDataIntervalBox->setToolTip(QString("Interval for auxilliary data readings (e.g., flows, pressure, etc.)"));
    p_auxDataIntervalBox->setValue(get(auxInterval,300));
    registerGetter(auxInterval,p_auxDataIntervalBox,&QSpinBox::value);
    fl2->addRow("Aux Data Interval",p_auxDataIntervalBox);

    p_backupBox = new QSpinBox(this);
    p_backupBox->setRange(0,100);
    p_backupBox->setSpecialValueText("Disabled");
    p_backupBox->setSuffix(QString(" hour"));
    p_backupBox->setToolTip(QString("Interval for autosaving."));
    p_backupBox->setValue(get(backup,0));
    registerGetter(backup,p_backupBox,&QSpinBox::value);
    fl2->addRow("Backup Interval",p_backupBox);


    for(int i=0; i<fl2->rowCount(); ++i)
    {
        auto lbl = dynamic_cast<QLabel*>(fl2->itemAt(i,QFormLayout::LabelRole)->widget());
        if(lbl != nullptr)
        {
            lbl->setAlignment(Qt::AlignRight|Qt::AlignCenter);
            lbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Expanding);
        }
    }
    sgb->setLayout(fl2);


    auto *hbl = new QHBoxLayout();
    hbl->addWidget(p_ftmw);

#ifndef BC_LIF
    p_ftmw->setChecked(true);
    p_ftmw->setCheckable(false);
#else
    p_lif = new QGroupBox(QString("LIF"),this);
    p_lif->setCheckable(true);
    p_lif->setChecked(get(lif,false));
    registerGetter(lif,p_lif,&QGroupBox::isChecked);
    connect(p_lif,&QGroupBox::toggled,this,&ExperimentTypePage::typeChanged);

    auto lvbl = new QVBoxLayout;
    p_lif->setLayout(lvbl);

    auto dlg = new QGroupBox("Delay",this);
    auto dl = new QGridLayout;
    dlg->setLayout(dl);

    p_dStartBox = new QDoubleSpinBox(this);
    p_dStartBox->setDecimals(3);
    p_dStartBox->setKeyboardTracking(false);
    p_dStartBox->setRange(0,100000.0);
    p_dStartBox->setSuffix(QString(" ").append(BC::Unit::us));
    p_dStartBox->setValue(get(lifDelayStart,p_dStartBox->minimum()));
    registerGetter(lifDelayStart,p_dStartBox,&QDoubleSpinBox::value);
    dl->addWidget(new QLabel("Start"),0,0,Qt::AlignRight);
    dl->addWidget(p_dStartBox,0,1);

    auto range = p_dStartBox->maximum() - p_dStartBox->minimum();

    p_dStepBox = new QDoubleSpinBox(this);
    p_dStepBox->setDecimals(3);
    p_dStepBox->setKeyboardTracking(false);
    p_dStepBox->setRange(-range,range);
    p_dStepBox->setSuffix(QString(" ").append(BC::Unit::us));
    p_dStepBox->setValue(get(lifDelayStep,0.0));
    registerGetter(lifDelayStep,p_dStepBox,&QDoubleSpinBox::value);
    dl->addWidget(new QLabel("Step"),0,2,Qt::AlignRight);
    dl->addWidget(p_dStepBox,0,3);

    p_dEndBox = new QDoubleSpinBox(this);
    p_dEndBox->setDecimals(3);
    p_dEndBox->setRange(0,100000.0);
    p_dEndBox->setSuffix(QString(" ").append(BC::Unit::us));
    p_dEndBox->setReadOnly(true);
    p_dEndBox->setButtonSymbols(QAbstractSpinBox::NoButtons);
    dl->addWidget(new QLabel("End"),1,0,Qt::AlignRight);
    dl->addWidget(p_dEndBox,1,1);

    p_dNumStepsBox = new QSpinBox(this);
    p_dNumStepsBox->setMinimum(1);
    p_dNumStepsBox->setKeyboardTracking(false);
    p_dNumStepsBox->setValue(get(lifDelayPoints,1));
    registerGetter(lifDelayPoints,p_dNumStepsBox,&QSpinBox::value);
    dl->addWidget(new QLabel("Points"),1,2,Qt::AlignRight);
    dl->addWidget(p_dNumStepsBox,1,3);

    lvbl->addWidget(dlg);

    auto llg = new QGroupBox("Laser",this);
    auto ll = new QGridLayout;
    llg->setLayout(ll);

    SettingsStorage lset(BC::Key::hwKey(BC::Key::LifLaser::key,0),SettingsStorage::Hardware);

    p_lStartBox = new QDoubleSpinBox(this);
    p_lStartBox->setDecimals(lset.get(BC::Key::LifLaser::decimals,2));
    p_lStartBox->setKeyboardTracking(false);
    p_lStartBox->setRange(lset.get(BC::Key::LifLaser::minPos,250.0),lset.get(BC::Key::LifLaser::maxPos,2000.0));
    p_lStartBox->setSuffix(QString(" ").append(lset.get(BC::Key::LifLaser::units,QString("nm"))));
    p_lStartBox->setValue(get(lifLaserStart,p_lStartBox->minimum()));
    registerGetter(lifLaserStart,p_lStartBox,&QDoubleSpinBox::value);
    ll->addWidget(new QLabel("Start"),0,0,Qt::AlignRight);
    ll->addWidget(p_lStartBox,0,1);

    range = p_lStartBox->maximum() - p_lStartBox->minimum();

    p_lStepBox = new QDoubleSpinBox(this);
    p_lStepBox->setKeyboardTracking(false);
    p_lStepBox->setDecimals(lset.get(BC::Key::LifLaser::decimals,2));
    p_lStepBox->setRange(-range,range);
    p_lStepBox->setSuffix(QString(" ").append(lset.get(BC::Key::LifLaser::units,QString("nm"))));
    p_lStepBox->setValue(get(lifLaserStep,0.0));
    registerGetter(lifLaserStep,p_lStepBox,&QDoubleSpinBox::value);
    ll->addWidget(new QLabel("Step"),0,2,Qt::AlignRight);
    ll->addWidget(p_lStepBox,0,3);

    p_lEndBox = new QDoubleSpinBox(this);
    p_lEndBox->setDecimals(3);
    p_lEndBox->setRange(lset.get(BC::Key::LifLaser::minPos,250.0),lset.get(BC::Key::LifLaser::maxPos,2000.0));
    p_lEndBox->setSuffix(QString(" ").append(lset.get(BC::Key::LifLaser::units,QString("nm"))));
    p_lEndBox->setReadOnly(true);
    p_lEndBox->setButtonSymbols(QAbstractSpinBox::NoButtons);
    ll->addWidget(new QLabel("End"),1,0,Qt::AlignRight);
    ll->addWidget(p_lEndBox,1,1);

    p_lNumStepsBox = new QSpinBox(this);
    p_lNumStepsBox->setKeyboardTracking(false);
    p_lNumStepsBox->setMinimum(1);
    p_lNumStepsBox->setValue(get(lifLaserPoints,1));
    registerGetter(lifLaserPoints,p_lNumStepsBox,&QSpinBox::value);
    ll->addWidget(new QLabel("Points"),1,2,Qt::AlignRight);
    ll->addWidget(p_lNumStepsBox,1,3);

    lvbl->addWidget(llg);

    auto optvbl = new QGroupBox("Options");
    auto ofl = new QFormLayout;
    optvbl->setLayout(ofl);

    p_orderBox = new QComboBox(this);
    auto t2 = QMetaEnum::fromType<LifConfig::LifScanOrder>();
    num = t2.keyCount();
    for(int i=0; i<num; ++i)
        p_orderBox->addItem(QString(t2.key(i)),
                            static_cast<LifConfig::LifScanOrder>(t2.value(i)));
    p_orderBox->setCurrentIndex(p_orderBox->findData(get(lifOrder,LifConfig::DelayFirst)));
    registerGetter(lifOrder,std::function<LifConfig::LifScanOrder()>([=](){
        return p_orderBox->currentData().value<LifConfig::LifScanOrder>();}
    ));
    ofl->addRow(QString("Scan Order"),p_orderBox);

    p_completeModeBox = new QComboBox(this);
    auto t3 = QMetaEnum::fromType<LifConfig::LifCompleteMode>();
    num = t3.keyCount();
    for(int i=0; i<num; i++)
        p_completeModeBox->addItem(QString(t3.key(i)),static_cast<LifConfig::LifCompleteMode>(t3.value(i)));
    p_completeModeBox->setCurrentIndex(p_completeModeBox->findData(get(lifCompleteMode,LifConfig::StopWhenComplete)));
    registerGetter(lifCompleteMode,std::function<LifConfig::LifCompleteMode()>([=](){
       return p_completeModeBox->currentData().value<LifConfig::LifCompleteMode>();
    }));
    ofl->addRow("Complete Mode",p_completeModeBox);
    p_completeModeBox->setEnabled(p_ftmw->isChecked());

    p_flBox = new QCheckBox(this);
    p_flBox->setChecked(get(lifFlashlampDisable,true));
    ofl->addRow("Auto Disable Flashlamp",p_flBox);

    lvbl->addWidget(optvbl);
    lvbl->addSpacerItem(new QSpacerItem(1,1));


    updateLifRanges();

    connect(p_dStartBox,qOverload<double>(&QDoubleSpinBox::valueChanged),this,&ExperimentTypePage::updateLifRanges);
    connect(p_dStepBox,qOverload<double>(&QDoubleSpinBox::valueChanged),this,&ExperimentTypePage::updateLifRanges);
    connect(p_dNumStepsBox,qOverload<int>(&QSpinBox::valueChanged),this,&ExperimentTypePage::updateLifRanges);
    connect(p_lStartBox,qOverload<double>(&QDoubleSpinBox::valueChanged),this,&ExperimentTypePage::updateLifRanges);
    connect(p_lStepBox,qOverload<double>(&QDoubleSpinBox::valueChanged),this,&ExperimentTypePage::updateLifRanges);
    connect(p_lNumStepsBox,qOverload<int>(&QSpinBox::valueChanged),this,&ExperimentTypePage::updateLifRanges);

    hbl->addWidget(p_lif);

    SettingsStorage s(BC::Key::hw);
    for(uint i=0; i<s.getArraySize(BC::Key::allHw); i++)
    {
        auto hwk = getArrayValue(BC::Key::allHw,i,BC::Key::HW::key,QString(""));
        auto l = hwk.split(BC::Key::hwIndexSep);
        if(!l.isEmpty())
        {
            if(l.constFirst() == BC::Key::PGen::key)
            {
                d_hasPGen = true;
                break;
            }
        }
    }
#endif

    auto *vbl = new QVBoxLayout();
    vbl->addLayout(hbl,1);
    vbl->addWidget(sgb);

    setLayout(vbl);

    if(p_exp->d_number > 0)
    {

#ifdef BC_LIF
        p_ftmw->setChecked(p_exp->ftmwEnabled());
        p_lif->setChecked(p_exp->lifEnabled());
#endif
        if(p_exp->ftmwEnabled())
        {
            auto type = p_exp->ftmwConfig()->d_type;
            p_ftmwTypeBox->setCurrentIndex(p_ftmwTypeBox->findData(QVariant::fromValue(type)));
            p_phaseCorrectionBox->setChecked(p_exp->ftmwConfig()->d_phaseCorrectionEnabled);
            p_chirpScoringBox->setChecked(p_exp->ftmwConfig()->d_chirpScoringEnabled);
            p_thresholdBox->setValue(p_exp->ftmwConfig()->d_chirpRMSThreshold);
            p_chirpOffsetBox->setValue(p_exp->ftmwConfig()->d_chirpOffsetUs);

            switch(type) {
            case FtmwConfig::Target_Shots:
            case FtmwConfig::Peak_Up:
            case FtmwConfig::DR_Scan:
            case FtmwConfig::LO_Scan:
                p_ftmwShotsBox->setValue(p_exp->ftmwConfig()->d_objective);
                break;
            case FtmwConfig::Target_Duration:
                p_ftmwTargetDurationBox->setValue(p_exp->ftmwConfig()->d_objective);
                break;
            default:
                break;
            }

        }


        p_backupBox->setValue(p_exp->d_backupIntervalHours);
        p_auxDataIntervalBox->setValue(p_exp->d_timeDataInterval);
    }

    connect(p_ftmwTypeBox,&QComboBox::currentTextChanged,this,&ExperimentTypePage::configureUI);
    connect(p_chirpScoringBox,&QCheckBox::toggled,this,&ExperimentTypePage::configureUI);
    connect(p_phaseCorrectionBox,&QCheckBox::toggled,this,&ExperimentTypePage::configureUI);

    using namespace std::chrono;
    d_timerId = startTimer(5s);
}

bool ExperimentTypePage::ftmwEnabled() const
{
    return !p_ftmw->isCheckable() || p_ftmw->isChecked();
}

FtmwConfig::FtmwType ExperimentTypePage::getFtmwType() const
{
    return p_ftmwTypeBox->currentData().value<FtmwConfig::FtmwType>();
}

#ifdef BC_LIF
bool ExperimentTypePage::lifEnabled() const
{
    return p_lif->isChecked();
}
#endif

void ExperimentTypePage::initialize()
{
    configureUI();
}

bool ExperimentTypePage::validate()
{
    if(!p_ftmw->isCheckable())
        return true;

    bool out = p_ftmw->isChecked();
#ifdef BC_LIF
    out = out || p_lif->isChecked();

    p_completeModeBox->setEnabled(p_ftmw->isChecked());
    if(!out)
        emit error("Either FTMW or LIF must be enabled.");

    if(!d_hasPGen && p_dNumStepsBox->value() > 1)
    {
        out = false;
        emit error("A pulse generator is required to step the LIF delay.");
    }
#endif

    return out;
}

void ExperimentTypePage::apply()
{
    auto e = p_exp;

#ifdef BC_LIF
     if(p_lif->isChecked())
     {
         e->enableLif();

         e->lifConfig()->d_delayStartUs = p_dStartBox->value();
         e->lifConfig()->d_delayStepUs = p_dStepBox->value();
         e->lifConfig()->d_delayPoints = p_dNumStepsBox->value();

         e->lifConfig()->d_laserPosStart = p_lStartBox->value();
         e->lifConfig()->d_laserPosStep = p_lStepBox->value();
         e->lifConfig()->d_laserPosPoints = p_lNumStepsBox->value();

         e->lifConfig()->d_completeMode = p_completeModeBox->currentData().value<LifConfig::LifCompleteMode>();
         e->lifConfig()->d_order = p_orderBox->currentData().value<LifConfig::LifScanOrder>();
         e->lifConfig()->d_disableFlashlamp = p_flBox->isChecked();
     }
     else
         e->disableLif();
#endif

     if(p_ftmw->isChecked() || !p_ftmw->isCheckable())
     {
         RfConfig cfg;

         SettingsStorage s(BC::Key::hwKey(BC::Key::FtmwScope::ftmwScope,0),SettingsStorage::Hardware);
         auto sk = s.get(BC::Key::HW::subKey,BC::Key::Comm::hwVirtual);
         FtmwDigitizerConfig ftc{sk};

         if(e->d_number > 0 && e->ftmwEnabled())
         {
             cfg = e->ftmwConfig()->d_rfConfig;
             ftc = e->ftmwConfig()->scopeConfig();
         }

         auto type = p_ftmwTypeBox->currentData().value<FtmwConfig::FtmwType>();
         auto ftmw = e->enableFtmw(type);

         ftmw->d_objective = p_ftmwShotsBox->value();
         if(type == FtmwConfig::Target_Duration)
             ftmw->d_objective = p_ftmwTargetDurationBox->value();
         ftmw->d_chirpScoringEnabled = p_chirpScoringBox->isChecked();
         ftmw->d_chirpRMSThreshold = p_thresholdBox->value();
         ftmw->d_phaseCorrectionEnabled = p_phaseCorrectionBox->isChecked();
         if(e->d_number > 0 && e->ftmwEnabled())
         {
             ftmw->d_rfConfig = cfg;
             ftmw->scopeConfig() = ftc;
         }
         if(p_chirpOffsetBox->value() >= 0.0)
             ftmw->d_chirpOffsetUs = p_chirpOffsetBox->value();
     }
     else
         e->disableFtmw();

     e->d_backupIntervalHours = p_backupBox->value();
     e->d_timeDataInterval = p_auxDataIntervalBox->value();
}

void ExperimentTypePage::configureUI()
{
    auto type = p_ftmwTypeBox->currentData().value<FtmwConfig::FtmwType>();

    switch(type)
    {
    case FtmwConfig::Target_Duration:
        p_ftmwShotsBox->setEnabled(false);
        p_ftmwTargetDurationBox->setEnabled(true);
        p_endTimeLabel->setEnabled(true);
        break;
    case FtmwConfig::Forever:
        p_ftmwShotsBox->setEnabled(false);
        p_ftmwTargetDurationBox->setEnabled(false);
        p_endTimeLabel->setEnabled(false);
        break;
    case FtmwConfig::LO_Scan:
    case FtmwConfig::DR_Scan:
        p_ftmwShotsBox->setEnabled(false);
        p_ftmwTargetDurationBox->setEnabled(false);
        p_endTimeLabel->setEnabled(false);
        break;
    default:
        p_ftmwShotsBox->setEnabled(true);
        p_ftmwTargetDurationBox->setEnabled(false);
        p_endTimeLabel->setEnabled(false);
        break;
    }

    p_chirpOffsetBox->setEnabled(p_phaseCorrectionBox->isChecked() || p_chirpScoringBox->isChecked());
    p_thresholdBox->setEnabled(p_chirpScoringBox->isChecked());
}

#ifdef BC_LIF
void ExperimentTypePage::updateLifRanges()
{
    auto maxdSteps = 1;
    if(p_dStepBox->value() < -1e-14)
        maxdSteps = static_cast<int>(floor((p_dEndBox->minimum() - p_dStartBox->value())/p_dStepBox->value()))+1;
    else if(p_dStepBox->value() > 1e-14)
        maxdSteps = static_cast<int>(floor((p_dEndBox->maximum() - p_dStartBox->value())/p_dStepBox->value()))+1;
    else
        maxdSteps = 1;
    p_dNumStepsBox->blockSignals(true);
    p_dNumStepsBox->setMaximum(qMax(1,maxdSteps));
    p_dNumStepsBox->blockSignals(false);

    auto dEnd = p_dStartBox->value() + (p_dNumStepsBox->value()-1)*p_dStepBox->value();
    p_dEndBox->setValue(dEnd);

    auto maxlSteps = 1;
    if(p_lStepBox->value() < -1e-14)
        maxlSteps = static_cast<int>(floor((p_lEndBox->minimum() - p_lStartBox->value())/p_lStepBox->value()))+1;
    else if(p_lStepBox->value() > 1e-14)
        maxlSteps = static_cast<int>(floor((p_lEndBox->maximum() - p_lStartBox->value())/p_lStepBox->value()))+1;
    else
        maxlSteps = 1;
    p_lNumStepsBox->blockSignals(true);
    p_lNumStepsBox->setMaximum(qMax(1,maxlSteps));
    p_lNumStepsBox->blockSignals(false);

    auto lEnd = p_lStartBox->value() + (p_lNumStepsBox->value()-1)*p_lStepBox->value();
    p_lEndBox->setValue(lEnd);

    p_orderBox->setDisabled(p_lNumStepsBox->value() == 1 || p_dNumStepsBox->value() == 1);
}
#endif

void ExperimentTypePage::updateLabel()
{
    auto dt = QDateTime::currentDateTime().addSecs(p_ftmwTargetDurationBox->value()*60);
    p_endTimeLabel->setText(d_endText.arg(dt.toString("yyyy-MM-dd h:mm AP")));
}

void ExperimentTypePage::timerEvent(QTimerEvent *event)
{
    if(event->timerId() == d_timerId)
    {
        updateLabel();
        event->accept();
    }
}
