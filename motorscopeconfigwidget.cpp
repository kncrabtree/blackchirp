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

    ui->triggerChannelSpinBox->setMinimum(s.value(QString("minTriggerChannel"),1).toInt());
    ui->triggerChannelSpinBox->setMaximum(s.value(QString("maxTriggerChannel"),2).toInt());

    ui->verticalRangeDoubleSpinBox->setMinimum(s.value(QString("minVerticalScale"),0.02).toDouble());
    ui->verticalRangeDoubleSpinBox->setMaximum(s.value(QString("maxVerticalScale"),20).toDouble());

    ui->recordLengthSpinBox->setMinimum(s.value(QString("minRecordLength"),1).toInt());
    ui->recordLengthSpinBox->setMaximum(s.value(QString("maxRecordLength"),32e6).toInt());

    ui->sampleIntervalDoubleSpinBox->setMinimum(1.0*1e6/(s.value(QString("maxSampleRate"), 1.0*1e9/16.0).toDouble()));
    ui->sampleIntervalDoubleSpinBox->setMaximum(1.0*1e6/(s.value(QString("minSampleRate"), 1.0/69.0).toDouble()));

    //ui->recordTimeDoubleSpinBox->setMinimum((static_cast<double>(s.value(QString("minRecordLength"),1).toInt())-1)*1e6/(s.value(QString("minSampleRate"), 1/69)).toDouble()*1e-3);
    //ui->recordTimeDoubleSpinBox->setMaximum((static_cast<double>(s.value(QString("maxRecordLength"),32e6).toInt())-1)*1e6/(s.value(QString("maxSampleRate"), 1e9/69)).toDouble()*1e-3);
    ui->recordTimeDoubleSpinBox->setValue((static_cast<double>(ui->recordLengthSpinBox->value())-1)*ui->sampleIntervalDoubleSpinBox->value()*1e-3);

    ui->triggerDirectionComboBox->setCurrentIndex(s.value(QString("slope"),BlackChirp::ScopeTriggerSlope::RisingEdge).toUInt());
    s.endGroup();
    s.endGroup();

    ui->recordTimeDoubleSpinBox->setSuffix(QString::fromUtf16(u" μs"));

    auto dvc = static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged);
    auto ivc = static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged);
    connect(ui->sampleIntervalDoubleSpinBox,dvc,[=](double t){ui->sampleRateDoubleSpinBox->setValue(1e9/t);});
    //connect(ui->sampleRateDoubleSpinBox_2,dvc,[=](double t){ui->sampleRateDoubleSpinBox->setValue(1e9/t);});
    connect(ui->recordLengthSpinBox,ivc,[=](int t){ui->recordTimeDoubleSpinBox->setValue(static_cast<double>(t-1)*ui->sampleIntervalDoubleSpinBox->value()*1e-3);});
    connect(ui->sampleIntervalDoubleSpinBox,dvc,[=](double t){ui->recordTimeDoubleSpinBox->setValue(t*(static_cast<double>(ui->recordLengthSpinBox->value())-1)*1e-3);});

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
    ui->sampleIntervalDoubleSpinBox->setValue(1.0/sc.sampleRate*1.0e9);
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
