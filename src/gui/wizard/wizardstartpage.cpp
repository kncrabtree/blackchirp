#include "wizardstartpage.h"

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QFormLayout>
#include <QCheckBox>
#include <QSpinBox>
#include <QGroupBox>
#include <QDoubleSpinBox>
#include <QComboBox>
#include <QLabel>
#include <QMetaEnum>

#include <gui/wizard/experimentwizard.h>

using namespace BC::Key::WizStart;

WizardStartPage::WizardStartPage(QWidget *parent) :
    ExperimentWizardPage(key,parent)
{
    setTitle(QString("Configure Experiment"));
    setSubTitle(QString("Choose which type(s) of experiment you wish to perform."));

    QFormLayout *fl = new QFormLayout(this);

    p_ftmw = new QGroupBox(QString("FTMW"),this);
    p_ftmw->setCheckable(true);
    p_ftmw->setChecked(get(ftmw,true));
    registerGetter(ftmw,p_ftmw,&QGroupBox::isChecked);
    connect(p_ftmw,&QGroupBox::toggled,this,&WizardStartPage::completeChanged);

    p_ftmwTypeBox = new QComboBox(this);
    auto t = QMetaEnum::fromType<FtmwConfig::FtmwType>();
    auto num = t.keyCount();
    for(int i=0; i<num; ++i)
        p_ftmwTypeBox->addItem(QString(t.key(i)).replace(QChar('_'),QChar(' ')),
                               static_cast<FtmwConfig::FtmwType>(t.value(i)));

    p_ftmwTypeBox->setCurrentIndex(get(ftmwType,0));
    registerGetter(ftmwType,p_ftmwTypeBox,&QComboBox::currentIndex);
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
    fl->addRow("Duration",p_ftmwTargetDurationBox);

    p_endTimeLabel = new QLabel;
    p_endTimeLabel->setAlignment(Qt::AlignLeft|Qt::AlignVCenter);
    p_endTimeLabel->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Expanding);
    auto dt = QDateTime::currentDateTime().addSecs(p_ftmwTargetDurationBox->value()*60);
    p_endTimeLabel->setText(d_endText.arg(dt.toString("yyyy-MM-dd h:mm AP")));
    fl->addRow(p_endTimeLabel);
    connect(p_ftmwTargetDurationBox,qOverload<int>(&QSpinBox::valueChanged),this,&WizardStartPage::updateLabel);

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
                lbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Expanding);
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

    p_autosaveBox = new QSpinBox(this);
    p_autosaveBox->setRange(0,100);
    p_autosaveBox->setSpecialValueText("Disabled");
    p_autosaveBox->setSuffix(QString(" hour"));
    p_autosaveBox->setToolTip(QString("Interval for autosaving."));
    p_autosaveBox->setValue(get(autosave,0));
    registerGetter(autosave,p_autosaveBox,&QSpinBox::value);
    fl2->addRow("Autosave Interval",p_autosaveBox);


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



#ifndef BC_LIF
    p_ftmw->setChecked(true);
#ifndef BC_MOTOR
    p_ftmw->setCheckable(false);
#endif
#else
    p_lif = new QGroupBox(QString("LIF"),this);
    p_lif->setCheckable(true);
    connect(p_lif,&QGroupBox::toggled,this,&WizardStartPage::completeChanged);
#endif

#ifdef BC_MOTOR
    p_motor = new QGroupBox(QString("Motor Scan"),this);
    p_motor->setCheckable(true);
    connect(p_motor,&QGroupBox::toggled,this,&WizardStartPage::completeChanged);
    connect(p_motor,&QGroupBox::toggled,[=](bool ch){
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

    auto *hbl = new QHBoxLayout();
    hbl->addWidget(p_ftmw);
#ifdef BC_LIF
    hbl->addWidget(p_lif);
#endif
#ifdef BC_MOTOR
    hbl->addWidget(p_motor);
#endif

    auto *vbl = new QVBoxLayout();
    vbl->addLayout(hbl,1);
    vbl->addWidget(sgb);

    setLayout(vbl);

    connect(p_ftmwTypeBox,&QComboBox::currentTextChanged,this,&WizardStartPage::configureUI);
    connect(p_chirpScoringBox,&QCheckBox::toggled,this,&WizardStartPage::configureUI);
    connect(p_phaseCorrectionBox,&QCheckBox::toggled,this,&WizardStartPage::configureUI);

    using namespace std::chrono;
    d_timerId = startTimer(5s);
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
    if(!p_ftmw->isCheckable())
        return true;

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

    if(e->d_number > 0)
    {

#ifdef BC_LIF
        p_ftmw->setChecked(e->ftmwEnabled());
        p_lif->setChecked(e->lifConfig().isEnabled());
#endif

#ifdef BC_MOTOR
        if(e->motorScan().isEnabled())
        {
            p_motor->setChecked(true);
            p_ftmw->setEnabled(false);
            p_ftmw->setChecked(false);
#ifdef BC_LIF
            p_lif->setChecked(false);
            p_lif->setEnabled(false);
#endif
        }
        else
        {
            p_motor->setChecked(false);
        }
#endif

        if(e->ftmwEnabled())
        {
            auto type = e->ftmwConfig()->d_type;
            p_ftmwTypeBox->setCurrentIndex(p_ftmwTypeBox->findData(QVariant::fromValue(type)));
            p_phaseCorrectionBox->setChecked(e->ftmwConfig()->d_phaseCorrectionEnabled);
            p_chirpScoringBox->setChecked(e->ftmwConfig()->d_chirpScoringEnabled);
            p_thresholdBox->setValue(e->ftmwConfig()->d_chirpRMSThreshold);
            ///TODO: use chirp offset!

            switch(type) {
            case FtmwConfig::Target_Shots:
            case FtmwConfig::Peak_Up:
            case FtmwConfig::DR_Scan:
            case FtmwConfig::LO_Scan:
                p_ftmwShotsBox->setValue(e->ftmwConfig()->d_objective);
                break;
            case FtmwConfig::Target_Duration:
                p_ftmwTargetDurationBox->setValue(e->ftmwConfig()->d_objective);
                break;
            default:
                break;
            }

        }


        p_autosaveBox->setValue(e->d_autoSaveIntervalHours);
        p_auxDataIntervalBox->setValue(e->d_timeDataInterval);
    }

    configureUI();
}

bool WizardStartPage::validatePage()
{
     auto e = getExperiment();

     if(p_ftmw->isChecked() || !p_ftmw->isCheckable())
     {
         auto type = p_ftmwTypeBox->currentData().value<FtmwConfig::FtmwType>();
         auto ftmw = e->enableFtmw(type);

         ftmw->d_objective = p_ftmwShotsBox->value();
         if(type == FtmwConfig::Target_Duration)
             ftmw->d_objective = p_ftmwTargetDurationBox->value();
         ftmw->d_chirpScoringEnabled = p_chirpScoringBox->isChecked();
         ftmw->d_chirpRMSThreshold = p_thresholdBox->value();
         ftmw->d_phaseCorrectionEnabled = p_phaseCorrectionBox->isChecked();
         ///TODO: use offset info!
     }



#ifdef BC_LIF
     e->setLifEnabled(p_lif->isChecked());
#endif

#ifdef BC_MOTOR
     e->setMotorEnabled(p_motor->isChecked());
#endif

     e->d_autoSaveIntervalHours = p_autosaveBox->value();
     e->d_timeDataInterval = p_auxDataIntervalBox->value();

     
     return true;

}

void WizardStartPage::configureUI()
{
    auto type = p_ftmwTypeBox->currentData().value<FtmwConfig::FtmwType>();

    switch(type)
    {
    case FtmwConfig::Target_Duration:
        p_ftmwShotsBox->setEnabled(false);
        p_ftmwTargetDurationBox->setEnabled(true);
        p_endTimeLabel->setEnabled(true);
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

void WizardStartPage::updateLabel()
{
    auto dt = QDateTime::currentDateTime().addSecs(p_ftmwTargetDurationBox->value()*60);
    p_endTimeLabel->setText(d_endText.arg(dt.toString("yyyy-MM-dd h:mm AP")));
}


void WizardStartPage::timerEvent(QTimerEvent *event)
{
    if(event->timerId() == d_timerId)
    {
        updateLabel();
        event->accept();
    }
}
