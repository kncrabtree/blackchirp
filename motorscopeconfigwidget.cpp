#include "motorscopeconfigwidget.h"
#include "ui_motorscopeconfigwidget.h"

MotorScopeConfigWidget::MotorScopeConfigWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::MotorScopeConfigWidget)
{
    ui->setupUi(this);

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(QString("motorScope"));
    s.beginGroup(QString("pico2206b"));
    ui->dataChannelSpinBox->setValue(s.value(QString("dataChannel"),1).toInt());
    ui->triggerChannelSpinBox->setValue(s.value(QString("triggerChannel"),2).toInt());
    ui->verticalRangeDoubleSpinBox->setValue(s.value(QString("verticalScale"),5.0).toDouble());
    ui->recordLengthSpinBox->setValue(s.value(QString("sampleRate"),100).toInt());
    ui->sampleRateDoubleSpinBox->setValue(s.value(QString("sampleRate"),500.0).toDouble());
    ui->triggerDirectionComboBox->setCurrentIndex(s.value(QString("slope"),BlackChirp::ScopeTriggerSlope::RisingEdge).toUInt());
    s.endGroup();
    s.endGroup();
}

MotorScopeConfigWidget::~MotorScopeConfigWidget()
{
    delete ui;
}

void MotorScopeConfigWidget::setFromConfig(const BlackChirp::MotorScopeConfig &sc)
{
    //initialize UI with settings in sc
    ui->dataChannelSpinBox->setValue(sc.dataChannel);
    ui->triggerChannelSpinBox->setValue(sc.triggerChannel);
    ui->verticalRangeDoubleSpinBox->setValue(sc.verticalScale);
    ui->sampleRateDoubleSpinBox->setValue(sc.sampleRate);
    ui->recordLengthSpinBox->setValue(sc.recordLength);
    ui->triggerDirectionComboBox->setCurrentIndex(static_cast<uint>(sc.slope));
}

BlackChirp::MotorScopeConfig MotorScopeConfigWidget::toConfig() const
{
    BlackChirp::MotorScopeConfig sc;
    sc.dataChannel = ui->dataChannelSpinBox->value();
    sc.triggerChannel = ui->triggerChannelSpinBox->value();
    sc.verticalScale = ui->verticalRangeDoubleSpinBox->value();
    sc.sampleRate = ui->sampleRateDoubleSpinBox->value();
    sc.recordLength = ui->recordLengthSpinBox->value();
    sc.slope = static_cast<BlackChirp::ScopeTriggerSlope>(ui->triggerDirectionComboBox->currentIndex());
    return sc;
}
