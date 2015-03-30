#include "ftmwconfigwidget.h"
#include "ui_ftmwconfigwidget.h"
#include <QSettings>
#include <QApplication>

FtmwConfigWidget::FtmwConfigWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::FtmwConfigWidget)
{
    ui->setupUi(this);

    ui->modeComboBox->addItem(QString("Target Shots"),QVariant::fromValue(FtmwConfig::TargetShots));
    ui->modeComboBox->addItem(QString("Target Time"),QVariant::fromValue(FtmwConfig::TargetTime));
    ui->modeComboBox->addItem(QString("Forever"),QVariant::fromValue(FtmwConfig::Forever));
    ui->modeComboBox->addItem(QString("Peak Up"),QVariant::fromValue(FtmwConfig::PeakUp));

    ui->sampleRateComboBox->addItem(QString("2 GS/s"),2e9);
    ui->sampleRateComboBox->addItem(QString("5 GS/s"),5e9);
    ui->sampleRateComboBox->addItem(QString("10 GS/s"),10e9);
    ui->sampleRateComboBox->addItem(QString("20 GS/s"),20e9);
    ui->sampleRateComboBox->addItem(QString("50 GS/s"),50e9);
    ui->sampleRateComboBox->addItem(QString("100 GS/s"),100e9);

    ui->sidebandComboBox->addItem(QString("Upper Sideband"),QVariant::fromValue(Fid::UpperSideband));
    ui->sidebandComboBox->addItem(QString("Lower Sideband"),QVariant::fromValue(Fid::LowerSideband));

    ui->triggerSlopeComboBox->addItem(QString("Rising Edge"),QVariant::fromValue(FtmwConfig::RisingEdge));
    ui->triggerSlopeComboBox->addItem(QString("Falling Edge"),QVariant::fromValue(FtmwConfig::FallingEdge));


    loadFromSettings();

    validateSpinboxes();

    connect(ui->ftmwEnabledCheckBox,&QCheckBox::toggled,this,&FtmwConfigWidget::configureUI);
    connect(ui->modeComboBox,static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged),this,&FtmwConfigWidget::configureUI);
    connect(ui->fastFrameEnabledCheckBox,&QCheckBox::toggled,this,&FtmwConfigWidget::configureUI);

    connect(ui->fIDChannelSpinBox,static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),this,&FtmwConfigWidget::validateSpinboxes);
    connect(ui->triggerChannelSpinBox,static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),this,&FtmwConfigWidget::validateSpinboxes);

}

FtmwConfigWidget::~FtmwConfigWidget()
{
    delete ui;
}

void FtmwConfigWidget::setFromConfig(const FtmwConfig config)
{
    blockSignals(true);

    ui->ftmwEnabledCheckBox->setChecked(config.isEnabled());

    setComboBoxIndex(ui->modeComboBox,config.type());
    ui->targetShotsSpinBox->setValue(config.targetShots());
    if(config.targetTime().isValid())
        ui->targetTimeDateTimeEdit->setDateTime(config.targetTime());
    ui->autosaveSpinBox->setValue(config.autoSaveShots());

    ui->loFrequencyDoubleSpinBox->setValue(config.loFreq());
    setComboBoxIndex(ui->sidebandComboBox,config.sideband());

    const FtmwConfig::ScopeConfig sc = config.scopeConfig();
    ui->fIDChannelSpinBox->setValue(sc.fidChannel);
    ui->verticalScaleDoubleSpinBox->setValue(sc.vScale);
    ui->triggerChannelSpinBox->setValue(sc.trigChannel);
    setComboBoxIndex(ui->triggerSlopeComboBox,sc.slope);
    setComboBoxIndex(ui->sampleRateComboBox,sc.sampleRate);
    ui->bytesPointSpinBox->setValue(sc.bytesPerPoint);
    ui->fastFrameEnabledCheckBox->setChecked(sc.fastFrameEnabled);
    ui->framesSpinBox->setValue(sc.numFrames);
    ui->summaryFrameCheckBox->setChecked(sc.summaryFrame);
    blockSignals(false);

    configureUI();
    validateSpinboxes();
}

FtmwConfig FtmwConfigWidget::getConfig() const
{
    FtmwConfig out;

    if(ui->ftmwEnabledCheckBox->isEnabled())
        out.setEnabled();

    out.setType(ui->modeComboBox->currentData().value<FtmwConfig::FtmwType>());
    out.setTargetShots(ui->targetShotsSpinBox->value());
    if(ui->targetTimeDateTimeEdit->dateTime() > QDateTime::currentDateTime().addSecs(60))
        out.setTargetTime(ui->targetTimeDateTimeEdit->dateTime());
    else
        out.setTargetTime(QDateTime::currentDateTime().addSecs(60));
    out.setAutoSaveShots(ui->autosaveSpinBox->value());

    out.setLoFreq(ui->loFrequencyDoubleSpinBox->value());
    out.setSideband(ui->sidebandComboBox->currentData().value<Fid::Sideband>());

    FtmwConfig::ScopeConfig sc;
    sc.fidChannel = ui->fIDChannelSpinBox->value();
    sc.vScale = ui->verticalScaleDoubleSpinBox->value();
    sc.trigChannel = ui->triggerChannelSpinBox->value();
    sc.slope = ui->triggerSlopeComboBox->currentData().value<FtmwConfig::ScopeTriggerSlope>();
    sc.sampleRate = ui->sampleRateComboBox->currentData().toDouble();
    sc.recordLength = ui->recordLengthSpinBox->value();
    sc.bytesPerPoint = ui->bytesPointSpinBox->value();
    sc.fastFrameEnabled = ui->fastFrameEnabledCheckBox->isChecked();
    sc.numFrames = ui->framesSpinBox->value();
    sc.summaryFrame = ui->summaryFrameCheckBox->isChecked();
    out.setScopeConfig(sc);

    return out;
}

void FtmwConfigWidget::loadFromSettings()
{
    blockSignals(true);
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());

    s.beginGroup(QString("lastFtmwConfig"));

    ui->ftmwEnabledCheckBox->setChecked(s.value(QString("ftmwEnabled"),true).toBool());

    ui->modeComboBox->setCurrentIndex(s.value(QString("mode"),0).toInt());
    ui->targetShotsSpinBox->setValue(s.value(QString("targetShots"),10000).toInt());
    ui->targetTimeDateTimeEdit->setMinimumDateTime(QDateTime::currentDateTime().addSecs(60));
    ui->targetTimeDateTimeEdit->setDateTime(QDateTime::currentDateTime().addSecs(3600));
    ui->targetTimeDateTimeEdit->setMaximumDateTime(QDateTime::currentDateTime().addSecs(2000000000));
    ui->targetTimeDateTimeEdit->setCurrentSection(QDateTimeEdit::HourSection);
    ui->autosaveSpinBox->setValue(s.value(QString("autosaveShots"),2500).toInt());

    ui->loFrequencyDoubleSpinBox->setValue(s.value(QString("loFreq"),41000.0).toDouble());
    ui->sidebandComboBox->setCurrentIndex(s.value(QString("sideband"),1).toInt());

    ui->fIDChannelSpinBox->setValue(s.value(QString("fidChannel"),1).toInt());
    ui->verticalScaleDoubleSpinBox->setValue(s.value(QString("vScale"),0.020).toDouble());
    ui->triggerChannelSpinBox->setValue(s.value(QString("triggerChannel"),4).toInt());
    ui->triggerSlopeComboBox->setCurrentIndex(s.value(QString("triggerSlope"),0).toInt());
    ui->sampleRateComboBox->setCurrentIndex(s.value(QString("sampleRate"),4).toInt());
    ui->recordLengthSpinBox->setValue(s.value(QString("recordLength"),750000).toInt());
    ui->bytesPointSpinBox->setValue(s.value(QString("bytesPerPoint"),1).toInt());
    ui->fastFrameEnabledCheckBox->setChecked(s.value(QString("fastFrame"),true).toBool());
    ui->framesSpinBox->setValue(s.value(QString("numFrames"),10).toInt());
    ui->summaryFrameCheckBox->setChecked(s.value(QString("summaryFrame"),false).toBool());

    s.endGroup();
    blockSignals(false);

    configureUI();


}

void FtmwConfigWidget::saveToSettings()
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());

    s.beginGroup(QString("lastFtmwConfig"));

    s.setValue(QString("ftmwEnabled"),ui->ftmwEnabledCheckBox->isChecked());

    s.setValue(QString("mode"),ui->modeComboBox->currentIndex());
    s.setValue(QString("targetShots"),ui->targetShotsSpinBox->value());
    s.setValue(QString("autosaveShots"),ui->autosaveSpinBox->value());

    s.setValue(QString("loFreq"),ui->loFrequencyDoubleSpinBox->value());
    s.setValue(QString("sideband"),ui->sidebandComboBox->currentIndex());

    s.setValue(QString("fidChannel"),ui->fIDChannelSpinBox->value());
    s.setValue(QString("vScale"),ui->verticalScaleDoubleSpinBox->value());
    s.setValue(QString("triggerChannel"),ui->triggerChannelSpinBox->value());
    s.setValue(QString("triggerSlope"),ui->triggerSlopeComboBox->currentIndex());
    s.setValue(QString("sampleRate"),ui->sampleRateComboBox->currentIndex());
    s.setValue(QString("recordLength"),ui->recordLengthSpinBox->value());
    s.setValue(QString("bytesPerPoint"),ui->bytesPointSpinBox->value());
    s.setValue(QString("fastFrame"),ui->fastFrameEnabledCheckBox->isChecked());
    s.setValue(QString("numFrames"),ui->framesSpinBox->value());
    s.setValue(QString("summaryFrame"),ui->summaryFrameCheckBox->isChecked());

    s.endGroup();
}

void FtmwConfigWidget::configureUI()
{
    blockSignals(true);
    ui->acqSettingsBox->setEnabled(ui->ftmwEnabledCheckBox->isChecked());
    ui->fidSettingsBox->setEnabled(ui->ftmwEnabledCheckBox->isChecked());
    ui->scopeSettingsBox->setEnabled(ui->ftmwEnabledCheckBox->isChecked());

    FtmwConfig::FtmwType type = ui->modeComboBox->currentData().value<FtmwConfig::FtmwType>();
    if(type == FtmwConfig::TargetTime)
    {
        ui->targetTimeDateTimeEdit->setEnabled(true);
        ui->targetShotsSpinBox->setEnabled(false);
    }
    else
    {
        ui->targetTimeDateTimeEdit->setEnabled(false);
        ui->targetShotsSpinBox->setEnabled(true);
    }

    if(type == FtmwConfig::Forever)
        ui->targetShotsSpinBox->setEnabled(false);

    if(type == FtmwConfig::PeakUp)
        ui->autosaveSpinBox->setEnabled(false);
    else
        ui->autosaveSpinBox->setEnabled(true);

    ui->framesSpinBox->setEnabled(ui->fastFrameEnabledCheckBox->isChecked());
    ui->summaryFrameCheckBox->setEnabled(ui->fastFrameEnabledCheckBox->isChecked());
    blockSignals(false);
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
            if(fabs(box->itemData(i).toDouble() - value.toDouble()) < 1.0)
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
