#include "motorscopeconfigwidget.h"
#include "ui_motorscopeconfigwidget.h"

MotorScopeConfigWidget::MotorScopeConfigWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::MotorScopeConfigWidget)
{
    ui->setupUi(this);

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(QString("motorScope"));
    s.beginGroup(s.value(QString("subKey"),QString("virtual")).toString());

    ui->dataChannelSpinBox->setMinimum(s.value(QString("minDataChannel"),1).toInt());
    ui->dataChannelSpinBox->setMaximum(s.value(QString("maxDataChannel"),2).toInt());
    ui->dataChannelSpinBox->setValue(s.value(QString("dataChannel"),1).toInt());

    ui->triggerChannelSpinBox->setMinimum(s.value(QString("minTriggerChannel"),1).toInt());
    ui->triggerChannelSpinBox->setMaximum(s.value(QString("maxTriggerChannel"),2).toInt());
    ui->triggerChannelSpinBox->setValue(s.value(QString("triggerChannel"),2).toInt());

    ui->verticalRangeDoubleSpinBox->setMinimum(s.value(QString("minVerticalScale"),0.02).toDouble());
    ui->verticalRangeDoubleSpinBox->setMaximum(s.value(QString("maxVerticalScale"),20).toDouble());
    ui->verticalRangeDoubleSpinBox->setValue(s.value(QString("verticalScale"),5.0).toDouble());

    ui->recordLengthSpinBox->setMinimum(s.value(QString("minRecordLength"),1  ).toInt());
    ui->recordLengthSpinBox->setMaximum(s.value(QString("maxRecordLength"),32e6).toInt()); //?
    ui->recordLengthSpinBox->setValue(s.value(QString("recordLength"),100).toInt());

    ui->sampleRateDoubleSpinBox->setMinimum(s.value(QString("minSampleRate"), 16).toDouble());
    ui->sampleRateDoubleSpinBox->setMaximum(s.value(QString("maxSampleRate"), 69e9).toDouble());
    ui->sampleRateDoubleSpinBox->setValue(s.value(QString("sampleRate"),500.0).toDouble());

    ui->recordTimeDoubleSpinBox->setMinimum((static_cast<double>(s.value(QString("minRecordLength"),1).toInt())-1)*s.value(QString("minSampleRate"), 16).toDouble()*1e-3);
    ui->recordTimeDoubleSpinBox->setMaximum((static_cast<double>(s.value(QString("maxRecordLength"),32e6).toInt())-1)*s.value(QString("maxSampleRate"), 69e9).toDouble()*1e-3);
    ui->recordTimeDoubleSpinBox->setValue((static_cast<double>(ui->recordLengthSpinBox->value())-1)*ui->sampleRateDoubleSpinBox->value()*1e-3);

    ui->triggerDirectionComboBox->setCurrentIndex(s.value(QString("slope"),BlackChirp::ScopeTriggerSlope::RisingEdge).toUInt());
    s.endGroup();
    s.endGroup();

    ui->recordTimeDoubleSpinBox->setSuffix(QString::fromUtf16(u" μs"));

    ui->sampleRateDoubleSpinBox_2->setValue(1e9/ui->sampleRateDoubleSpinBox->value());
    auto dvc = static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged);
    auto ivc = static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged);
    connect(ui->sampleRateDoubleSpinBox,dvc,[=](double t){ui->sampleRateDoubleSpinBox_2->setValue(1e9/t);});
    //connect(ui->sampleRateDoubleSpinBox_2,dvc,[=](double t){ui->sampleRateDoubleSpinBox->setValue(1e9/t);});
    connect(ui->recordLengthSpinBox,ivc,[=](int t){ui->recordTimeDoubleSpinBox->setValue(static_cast<double>(t-1)*ui->sampleRateDoubleSpinBox->value());});
    connect(ui->sampleRateDoubleSpinBox,dvc,[=](double t){ui->recordTimeDoubleSpinBox->setValue(t*(static_cast<double>(ui->recordLengthSpinBox->value())-1));});

//    //if the dataChannel and triggerChannel can't be the same.
//    auto vc = static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged);
//    connect(ui->dataChannelSpinBox,vc,[=](int t){ui->triggerChannelSpinBox->setValue(3-t);});
//    connect(ui->triggerChannelSpinBox,vc,[=](int t){ui->dataChannelSpinBox->setValue(3-t);});
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
