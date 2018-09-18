#include "wizardstartpage.h"

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QFormLayout>
#include <QCheckBox>
#include <QSpinBox>
#include <QGroupBox>
#include <QDateTimeEdit>
#include <QDoubleSpinBox>
#include <QComboBox>

#include "experimentwizard.h"

WizardStartPage::WizardStartPage(QWidget *parent) :
    ExperimentWizardPage(parent)
{
    setTitle(QString("Configure Experiment"));
    setSubTitle(QString("Choose which type(s) of experiment you wish to perform."));

    QFormLayout *fl = new QFormLayout(this);

    p_ftmw = new QGroupBox(QString("FTMW"),this);
    p_ftmw->setCheckable(true);
    connect(p_ftmw,&QGroupBox::toggled,this,&WizardStartPage::completeChanged);

    p_ftmwTypeBox = new QComboBox(this);
    p_ftmwTypeBox->addItem(QString("Target Shots"),QVariant::fromValue(BlackChirp::FtmwTargetShots));
    p_ftmwTypeBox->addItem(QString("Target Time"),QVariant::fromValue(BlackChirp::FtmwTargetTime));
    p_ftmwTypeBox->addItem(QString("Forever"),QVariant::fromValue(BlackChirp::FtmwForever));
    p_ftmwTypeBox->addItem(QString("Peak Up"),QVariant::fromValue(BlackChirp::FtmwPeakUp));
    p_ftmwTypeBox->setCurrentIndex(0);
    fl->addRow(QString("Type"),p_ftmwTypeBox);

    p_ftmwShotsBox = new QSpinBox(this);
    p_ftmwShotsBox->setRange(1,__INT_MAX__);
    p_ftmwShotsBox->setToolTip(QString("Number of FIDs to average.\n"
                                       "When this number is reached, the experiment ends in Target Shots mode, while an exponentially weighted moving average engages in Peak Up mode.\n\n"
                                       "If this box is disabled, it is either irrelevant or will be configured on a later page (e.g. in Multiple LO mode)."));
    p_ftmwShotsBox->setSingleStep(5000);
    fl->addRow(QString("Shots"),p_ftmwShotsBox);

    p_ftmwTargetTimeBox = new QDateTimeEdit(this);
    p_ftmwTargetTimeBox->setDisplayFormat(QString("yyyy-MM-dd h:mm:ss AP"));
    p_ftmwTargetTimeBox->setMaximumDateTime(QDateTime::currentDateTime().addSecs(__INT_MAX__));
    p_ftmwTargetTimeBox->setCurrentSection(QDateTimeEdit::HourSection);
    p_ftmwTargetTimeBox->setToolTip(QString("The time at which an experiment in Target Time mode will complete. If disabled, this setting is irrelevant."));
    fl->addRow(QString("Stop Time"),p_ftmwTargetTimeBox);

    p_phaseCorrectionBox = new QCheckBox(this);
    p_phaseCorrectionBox->setToolTip(QString("If checked, Blackchirp will optimize the autocorrelation of the chirp during the acquisition.\n\nFor this to work, the chirp must be part of the signal recorded by the digitizer."));
    fl->addRow(QString("Phase Correction"),p_phaseCorrectionBox);

    p_chirpScoringBox = new QCheckBox(this);
    p_chirpScoringBox->setToolTip(QString("If checked, Blackchirp will compare the RMS of the chirp in each new waveform with that of the current average chirp RMS.\nIf less than threshold*averageRMS, the FID will be rejected."));
    fl->addRow(QString("Chirp Scoring"),p_chirpScoringBox);

    p_thresholdBox = new QDoubleSpinBox(this);
    p_thresholdBox->setRange(0.0,1.0);
    p_thresholdBox->setSingleStep(0.05);
    p_thresholdBox->setValue(0.9);
    p_thresholdBox->setDecimals(3);
    fl->addRow(QString("Chirp Threshold"),p_thresholdBox);

    p_chirpOffsetBox = new QDoubleSpinBox(this);
    p_chirpOffsetBox->setRange(-0.00001,100.0);
    p_chirpOffsetBox->setDecimals(5);
    p_chirpOffsetBox->setSingleStep(0.1);
    p_chirpOffsetBox->setValue(-1.0);
    p_chirpOffsetBox->setSuffix(QString::fromUtf16(u" μs"));
    p_chirpOffsetBox->setSpecialValueText(QString("Automatic"));
    p_chirpOffsetBox->setToolTip(QString("The time at which the chirp starts (used for phase correction and chirp scoring).\n\nIf automatic, Blackchirp assumes the digitizer is triggered at the start of the protection pulse,\nand accounts for the digitizer trigger position."));
    fl->addRow(QString("Chirp Start"),p_chirpOffsetBox);

    p_ftmw->setLayout(fl);


    auto *sgb = new QGroupBox(QString("Common Settings"));
    p_auxDataIntervalBox = new QSpinBox(this);
    p_auxDataIntervalBox->setRange(5,__INT_MAX__);
    p_auxDataIntervalBox->setValue(300);
    p_auxDataIntervalBox->setSingleStep(300);
    p_auxDataIntervalBox->setSuffix(QString(" s"));
    p_auxDataIntervalBox->setToolTip(QString("Interval for auxilliary data readings (e.g., flows, pressure, etc.)"));

    p_snapshotBox = new QSpinBox(this);
    p_snapshotBox->setRange(1<<8,(1<<30)-1);
    p_snapshotBox->setValue(20000);
    p_snapshotBox->setSingleStep(5000);
    p_snapshotBox->setPrefix(QString("every "));
    p_snapshotBox->setSuffix(QString(" shots"));
    p_snapshotBox->setToolTip(QString("Interval for taking experiment snapshots (i.e., autosaving)."));

    auto *fl2 = new QFormLayout(this);
    fl2->addRow(QString("Time Data Interval"),p_auxDataIntervalBox);
    fl2->addRow(QString("Snapshot Interval"),p_snapshotBox);
    sgb->setLayout(fl2);



#ifndef BC_LIF

    p_ftmw->setChecked(true);
#ifndef BC_MOTOR
    p_ftmw->setEnabled(false);
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
        return startingFtmwPage();
#endif

    return startingFtmwPage();
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

    p_ftmwTypeBox->setCurrentIndex(p_ftmwTypeBox->findData(QVariant::fromValue(e.ftmwConfig().type())));
    p_ftmwShotsBox->setValue(e.ftmwConfig().targetShots());
    p_ftmwTargetTimeBox->setMinimumDateTime(QDateTime::currentDateTime().addSecs(60));
    p_ftmwTargetTimeBox->setDateTime(QDateTime::currentDateTime().addSecs(3600));
    p_phaseCorrectionBox->setChecked(e.ftmwConfig().isPhaseCorrectionEnabled());
    p_chirpScoringBox->setChecked(e.ftmwConfig().isChirpScoringEnabled());
    p_thresholdBox->setValue(e.ftmwConfig().chirpRMSThreshold());

    ///TODO: use chirp offset!

    p_snapshotBox->setValue(e.autoSaveShots());
    p_auxDataIntervalBox->setValue(e.timeDataInterval());

    configureUI();
}

bool WizardStartPage::validatePage()
{
    ///TODO: In the future, allow user to choose old experiment to repeat!
    ///Be sure to give user the options to use current pulse settings.
    /// Allow changing flow settings?
     auto e = getExperiment();

     auto ftc = e.ftmwConfig();
     ftc.setType(p_ftmwTypeBox->currentData().value<BlackChirp::FtmwType>());
     ftc.setTargetShots(p_ftmwShotsBox->value());
     ftc.setTargetTime(p_ftmwTargetTimeBox->dateTime());
     ftc.setChirpScoringEnabled(p_chirpScoringBox->isChecked());
     ftc.setChirpRMSThreshold(p_thresholdBox->value());
     ftc.setPhaseCorrectionEnabled(p_phaseCorrectionBox->isChecked());
     ///TODO: use offset info!

     e.setFtmwConfig(ftc);
     e.setFtmwEnabled(p_ftmw->isChecked());


#ifdef BC_LIF
     e.setLifEnabled(p_lif->isChecked());
#endif

#ifdef BC_MOTOR
     e.setMotorEnabled(p_motor->isChecked());
#endif

     e.setAutoSaveShotsInterval(p_snapshotBox->value());
     e.setTimeDataInterval(p_auxDataIntervalBox->value());

     emit experimentUpdate(e);
     return true;

}

void WizardStartPage::configureUI()
{
    auto type = p_ftmwTypeBox->currentData().value<BlackChirp::FtmwType>();
    switch(type)
    {
    case BlackChirp::FtmwForever:
    case BlackChirp::FtmwPeakUp:
    case BlackChirp::FtmwTargetShots:
        p_ftmwShotsBox->setEnabled(true);
        p_ftmwTargetTimeBox->setEnabled(false);
        break;
    case BlackChirp::FtmwTargetTime:
        p_ftmwShotsBox->setEnabled(false);
        p_ftmwTargetTimeBox->setEnabled(true);
        break;
    default:
        p_ftmwShotsBox->setEnabled(false);
        p_ftmwTargetTimeBox->setEnabled(false);
        break;
    }

    p_chirpOffsetBox->setEnabled(p_phaseCorrectionBox->isChecked() || p_chirpScoringBox->isChecked());
    p_thresholdBox->setEnabled(p_chirpScoringBox->isChecked());
}
