#include "ftmwconfigwidget.h"
#include "ui_ftmwconfigwidget.h"

#include <QSettings>
#include <QApplication>

#include "ftmwconfig.h"

FtmwConfigWidget::FtmwConfigWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::FtmwConfigWidget)
{
    ui->setupUi(this);

    ui->modeComboBox->addItem(QString("Target Shots"),QVariant::fromValue(BlackChirp::FtmwTargetShots));
    ui->modeComboBox->addItem(QString("Target Time"),QVariant::fromValue(BlackChirp::FtmwTargetTime));
    ui->modeComboBox->addItem(QString("Forever"),QVariant::fromValue(BlackChirp::FtmwForever));
    ui->modeComboBox->addItem(QString("Peak Up"),QVariant::fromValue(BlackChirp::FtmwPeakUp));

    ///TODO: Customize more UI settings according to Hardware limits for ftmwscope implementation
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());

    s.beginGroup(QString("ftmwscope"));
    s.beginGroup(s.value(QString("subKey"),QString("virtual")).toString());
    int size = s.beginReadArray(QString("sampleRates"));
    if(size > 0)
    {
        for(int i=0; i<size; i++)
        {
            s.setArrayIndex(i);
            QString str = s.value(QString("text"),QString("")).toString();
            double val = s.value(QString("val"),0.0).toDouble();
            ui->sampleRateComboBox->addItem(str,val);
        }
    }
    else
    {
        ui->sampleRateComboBox->addItem(QString("2 GS/s"),2e9);
        ui->sampleRateComboBox->addItem(QString("5 GS/s"),5e9);
        ui->sampleRateComboBox->addItem(QString("10 GS/s"),10e9);
        ui->sampleRateComboBox->addItem(QString("20 GS/s"),20e9);
        ui->sampleRateComboBox->addItem(QString("50 GS/s"),50e9);
        ui->sampleRateComboBox->addItem(QString("100 GS/s"),100e9);
    }
    s.endArray();

    s.endGroup();
    s.endGroup();

    ui->sidebandComboBox->addItem(QString("Upper Sideband"),QVariant::fromValue(BlackChirp::UpperSideband));
    ui->sidebandComboBox->addItem(QString("Lower Sideband"),QVariant::fromValue(BlackChirp::LowerSideband));

    ui->triggerSlopeComboBox->addItem(QString("Rising Edge"),QVariant::fromValue(BlackChirp::RisingEdge));
    ui->triggerSlopeComboBox->addItem(QString("Falling Edge"),QVariant::fromValue(BlackChirp::FallingEdge));

    ui->triggerDelayDoubleSpinBox->setSuffix(QString::fromUtf16(u" μs"));

    ui->targetTimeDateTimeEdit->setMinimumDateTime(QDateTime::currentDateTime().addSecs(60));
    ui->targetTimeDateTimeEdit->setDateTime(QDateTime::currentDateTime().addSecs(3600));
    ui->targetTimeDateTimeEdit->setMaximumDateTime(QDateTime::currentDateTime().addSecs(2000000000));
    ui->targetTimeDateTimeEdit->setCurrentSection(QDateTimeEdit::HourSection);

    setFromConfig(FtmwConfig::loadFromSettings());

    double loFreq = s.value(QString("rfConfig/loFreq"),0.0).toDouble();
    int sideband = s.value(QString("rfConfig/rxSidebandSign"),1).toInt();
    if(sideband > 0)
        ui->sidebandComboBox->setCurrentIndex(ui->sidebandComboBox->findData(QVariant::fromValue(BlackChirp::UpperSideband)));
    else
        ui->sidebandComboBox->setCurrentIndex(ui->sidebandComboBox->findData(QVariant::fromValue(BlackChirp::LowerSideband)));
    ui->loFrequencyDoubleSpinBox->setValue(loFreq);


    connect(ui->modeComboBox,static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged),this,&FtmwConfigWidget::configureUI);
    connect(ui->fastFrameEnabledCheckBox,&QCheckBox::toggled,this,&FtmwConfigWidget::configureUI);
    connect(ui->blockAverageCheckBox,&QCheckBox::toggled,this,&FtmwConfigWidget::configureUI);
    connect(ui->chirpScoringCheckBox,&QCheckBox::toggled,ui->chirpThresholdDoubleSpinBox,&QDoubleSpinBox::setEnabled);

    connect(ui->fIDChannelSpinBox,static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),this,&FtmwConfigWidget::validateSpinboxes);
    connect(ui->triggerChannelSpinBox,static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),this,&FtmwConfigWidget::validateSpinboxes);

}

FtmwConfigWidget::~FtmwConfigWidget()
{
    delete ui;
}

void FtmwConfigWidget::setFromConfig(const FtmwConfig config)
{
    ui->modeComboBox->blockSignals(true);
    setComboBoxIndex(ui->modeComboBox,qVariantFromValue(config.type()));
    ui->modeComboBox->blockSignals(false);

    ui->targetShotsSpinBox->setValue(config.targetShots());
    if(config.targetTime().isValid())
    {
        ui->targetTimeDateTimeEdit->setEnabled(true);
        ui->targetTimeDateTimeEdit->setDateTime(config.targetTime());
    }
    ui->phaseCorrectionCheckBox->setChecked(config.isPhaseCorrectionEnabled());

    ui->chirpScoringCheckBox->blockSignals(true);
    ui->chirpScoringCheckBox->setChecked(config.isChirpScoringEnabled());
    ui->chirpThresholdDoubleSpinBox->setValue(config.chirpRMSThreshold());
    ui->chirpThresholdDoubleSpinBox->setEnabled(config.isChirpScoringEnabled());
    ui->chirpScoringCheckBox->blockSignals(false);

    ui->loFrequencyDoubleSpinBox->setValue(config.loFreq());
    setComboBoxIndex(ui->sidebandComboBox,config.sideband());

    const BlackChirp::FtmwScopeConfig sc = config.scopeConfig();

    ui->fIDChannelSpinBox->blockSignals(true);
    ui->fIDChannelSpinBox->setValue(sc.fidChannel);
    ui->fIDChannelSpinBox->blockSignals(false);
    ui->verticalScaleDoubleSpinBox->setValue(sc.vScale);
    ui->triggerChannelSpinBox->blockSignals(true);
    ui->triggerChannelSpinBox->setValue(sc.trigChannel);
    ui->triggerChannelSpinBox->blockSignals(false);
    ui->triggerDelayDoubleSpinBox->setValue(sc.trigDelay*1e6);
    ui->triggerLevelDoubleSpinBox->setValue(sc.trigLevel);
    setComboBoxIndex(ui->triggerSlopeComboBox,qVariantFromValue(sc.slope));
    setComboBoxIndex(ui->sampleRateComboBox,sc.sampleRate);
    ui->recordLengthSpinBox->setValue(sc.recordLength);
    ui->bytesPointSpinBox->setValue(sc.bytesPerPoint);
    ui->fastFrameEnabledCheckBox->blockSignals(true);
    ui->fastFrameEnabledCheckBox->setChecked(sc.fastFrameEnabled);
    ui->fastFrameEnabledCheckBox->blockSignals(false);
    ui->framesSpinBox->setValue(sc.numFrames);
    ui->summaryFrameCheckBox->setChecked(sc.summaryFrame);
    ui->blockAverageCheckBox->setChecked(sc.blockAverageEnabled);
    ui->averagesSpinBox->setValue(sc.numAverages);


    configureUI();
    validateSpinboxes();
}

FtmwConfig FtmwConfigWidget::getConfig() const
{
    FtmwConfig out;

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());

    s.beginGroup(QString("ftmwscope"));
    s.beginGroup(s.value(QString("subKey"),QString("virtual")).toString());
    bool canSf = s.value(QString("canSummaryFrame"),false).toBool();
    s.endGroup();
    s.endGroup();

    out.setType(ui->modeComboBox->currentData().value<BlackChirp::FtmwType>());
    out.setTargetShots(ui->targetShotsSpinBox->value());
    if(ui->targetTimeDateTimeEdit->dateTime() > QDateTime::currentDateTime().addSecs(60))
        out.setTargetTime(ui->targetTimeDateTimeEdit->dateTime());
    else
        out.setTargetTime(QDateTime::currentDateTime().addSecs(60));
    out.setPhaseCorrectionEnabled(ui->phaseCorrectionCheckBox->isChecked());
    out.setChirpScoringEnabled(ui->chirpScoringCheckBox->isChecked());
    out.setChirpRMSThreshold(ui->chirpThresholdDoubleSpinBox->value());

    out.setLoFreq(ui->loFrequencyDoubleSpinBox->value());
    out.setSideband(ui->sidebandComboBox->currentData().value<BlackChirp::Sideband>());

    BlackChirp::FtmwScopeConfig sc;
    sc.fidChannel = ui->fIDChannelSpinBox->value();
    sc.vScale = ui->verticalScaleDoubleSpinBox->value();
    sc.trigChannel = ui->triggerChannelSpinBox->value();
    sc.trigDelay = ui->triggerDelayDoubleSpinBox->value()/1e6;
    sc.trigLevel = ui->triggerLevelDoubleSpinBox->value();
    sc.slope = ui->triggerSlopeComboBox->currentData().value<BlackChirp::ScopeTriggerSlope>();
    sc.sampleRate = ui->sampleRateComboBox->currentData().toDouble();
    sc.recordLength = ui->recordLengthSpinBox->value();
    sc.bytesPerPoint = ui->bytesPointSpinBox->value();
    sc.fastFrameEnabled = ui->fastFrameEnabledCheckBox->isChecked();
    if(sc.fastFrameEnabled && !canSf)
        sc.manualFrameAverage = true;
    sc.numFrames = ui->framesSpinBox->value();
    sc.summaryFrame = ui->summaryFrameCheckBox->isChecked();
    sc.blockAverageEnabled = ui->blockAverageCheckBox->isChecked();
    sc.numAverages = ui->averagesSpinBox->value();
    out.setScopeConfig(sc);

    return out;
}

void FtmwConfigWidget::lockFastFrame(const int nf)
{

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());

    ui->fastFrameEnabledCheckBox->blockSignals(true);
    ui->framesSpinBox->setValue(nf);
    ui->framesSpinBox->setEnabled(false);
    if(nf == 1)
    {
        ui->fastFrameEnabledCheckBox->setChecked(false);
        ui->summaryFrameCheckBox->setChecked(false);
        ui->summaryFrameCheckBox->setEnabled(false);
    }
    else
        ui->fastFrameEnabledCheckBox->setChecked(true);

    ui->fastFrameEnabledCheckBox->setEnabled(false);
    ui->fastFrameEnabledCheckBox->blockSignals(false);

    configureUI();
}

void FtmwConfigWidget::configureUI()
{

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());

    s.beginGroup(QString("ftmwscope"));
    s.beginGroup(s.value(QString("subKey"),QString("virtual")).toString());
    bool ba = s.value(QString("canBlockAverage"),false).toBool();
    bool ffba = s.value(QString("canBlockAndFastFrame"),false).toBool();
    bool ff = ui->fastFrameEnabledCheckBox->isChecked();
    s.endGroup();
    s.endGroup();

    if(!ba || (!ffba && ff))
    {
        ui->blockAverageCheckBox->setChecked(false);
        ui->blockAverageCheckBox->setEnabled(false);
        ui->averagesSpinBox->setValue(1);
        ui->averagesSpinBox->setEnabled(false);
    }
    else
    {
        ui->blockAverageCheckBox->setEnabled(true);
        ui->averagesSpinBox->setEnabled(true);
    }

    BlackChirp::FtmwType type = ui->modeComboBox->currentData().value<BlackChirp::FtmwType>();
    if(type == BlackChirp::FtmwTargetTime)
    {
        ui->targetTimeDateTimeEdit->setEnabled(true);
        ui->targetShotsSpinBox->setEnabled(false);
    }
    else
    {
        ui->targetTimeDateTimeEdit->setEnabled(false);
        ui->targetShotsSpinBox->setEnabled(true);
    }

    if(type == BlackChirp::FtmwForever)
        ui->targetShotsSpinBox->setEnabled(false);


    if(ui->fastFrameEnabledCheckBox->isEnabled())
    {
        ui->framesSpinBox->setEnabled(ui->fastFrameEnabledCheckBox->isChecked());
        ui->summaryFrameCheckBox->setEnabled(ui->fastFrameEnabledCheckBox->isChecked());
    }

}

void FtmwConfigWidget::validateSpinboxes()
{
    blockSignals(true);
    QObject *s = sender();
    if(s == nullptr)
        s = ui->triggerChannelSpinBox;

    QSpinBox *senderBox = dynamic_cast<QSpinBox*>(s);
    if(senderBox == nullptr)
        return;

    QSpinBox *otherBox;
    if(senderBox == ui->fIDChannelSpinBox)
        otherBox = ui->triggerChannelSpinBox;
    else
        otherBox = ui->fIDChannelSpinBox;

    if(senderBox->value() == otherBox->value())
    {
        if(otherBox->value() + 1 > senderBox->maximum())
            otherBox->setValue(otherBox->value() - 1);
        else
            senderBox->setValue(otherBox->value() + 1);
    }
    blockSignals(false);


}

void FtmwConfigWidget::setComboBoxIndex(QComboBox *box, QVariant value)
{
    for(int i=0; i<box->count(); i++)
    {
        if(box == ui->sampleRateComboBox)
        {
            if(qAbs(box->itemData(i).toDouble() - value.toDouble()) < 1.0)
            {
                box->setCurrentIndex(i);
                return;
            }
        }
        if(box->itemData(i) == value)
        {
            box->setCurrentIndex(i);
            return;
        }
    }
}
