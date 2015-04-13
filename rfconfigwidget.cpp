#include "rfconfigwidget.h"
#include "ui_rfconfigwidget.h"
#include <QSettings>
#include <QApplication>

RfConfigWidget::RfConfigWidget(double tx, double rx, QWidget *parent) :
    QWidget(parent), ui(new Ui::RfConfigWidget), d_valonTxFreq(tx), d_valonRxFreq(rx)
{
    ui->setupUi(this);

    loadFromSettings();

    auto vcint = static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged);
//    auto vcdbl = static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged);
    auto vccmbo = static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged);

    connect(ui->awgMultiplicationSpinBox,vcint,this,&RfConfigWidget::validate);
    connect(ui->valonTxMultiplicationSpinBox,vcint,this,&RfConfigWidget::validate);
    connect(ui->txSidebandComboBox,vccmbo,this,&RfConfigWidget::validate);
    connect(ui->totalMultiplicationSpinBox,vcint,this,&RfConfigWidget::validate);
    connect(ui->valonRxMultiplicationSpinBox,vcint,this,&RfConfigWidget::validate);
    connect(ui->rxSidebandComboBox,vccmbo,this,&RfConfigWidget::validate);

    connect(ui->txApplyButton,&QPushButton::clicked,this,[=](){ emit setValonTx( ui->valonTxFrequencyDoubleSpinBox->value()); });
    connect(ui->rxApplyButton,&QPushButton::clicked,this,[=]()
    {
        double d = ui->chirpLOFrequencyDoubleSpinBox->value()/static_cast<double>(ui->valonRxMultiplicationSpinBox->value());
        emit setValonRx(d);
    });


}

RfConfigWidget::~RfConfigWidget()
{
    delete ui;
}

void RfConfigWidget::loadFromSettings()
{
    blockSignals(true);
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());

    double valonMin = s.value(QString("valonSynth/minFreq"),500.0).toDouble();
    double valonMax = s.value(QString("valonSynth/maxFreq"),6000.0).toDouble();

    s.beginGroup(QString("chirpConfig"));

    ui->awgMultiplicationSpinBox->setValue(s.value(QString("awgMult"),1).toInt());
    ui->valonTxMultiplicationSpinBox->setValue(s.value(QString("txValonMult"),2).toInt());
    d_txSidebandSign = s.value(QString("txSidebandSign"),-1.0).toDouble();
    ui->txSidebandComboBox->setCurrentIndex(sidebandIndex(d_txSidebandSign));
    ui->totalMultiplicationSpinBox->setValue(s.value(QString("txMult"),4).toInt());
    ui->valonTxFrequencyDoubleSpinBox->setRange(valonMin,valonMax);
    ui->valonTxFrequencyDoubleSpinBox->setValue(d_valonTxFreq);

    d_currentRxMult = s.value(QString("rxValonMult"),8.0).toDouble();
    ui->valonRxMultiplicationSpinBox->setValue(s.value(QString("rxValonMult"),8).toInt());
    ui->chirpLOFrequencyDoubleSpinBox->setMinimum(valonMin*static_cast<double>(ui->valonRxMultiplicationSpinBox->value()));
    ui->chirpLOFrequencyDoubleSpinBox->setMaximum(valonMax*static_cast<double>(ui->valonRxMultiplicationSpinBox->value()));
    d_rxSidebandSign = s.value(QString("rxSidebandSign"),-1.0).toDouble();
    ui->rxSidebandComboBox->setCurrentIndex(sidebandIndex(d_rxSidebandSign));
    ui->chirpLOFrequencyDoubleSpinBox->setValue(d_valonRxFreq*static_cast<double>(ui->valonRxMultiplicationSpinBox->value()));
    ui->valonRxFrequencyDisplayLabel->setText(QString("%1 MHz").arg(d_valonRxFreq,0,'f',3));

    s.endGroup();
    blockSignals(false);

    validate();
}

void RfConfigWidget::txFreqUpdate(const double d)
{
    d_valonTxFreq = d;
    validate();
}

void RfConfigWidget::rxFreqUpdate(const double d)
{
    d_valonRxFreq = d;
    validate();
}

void RfConfigWidget::validate()
{
    blockSignals(true);
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());

    double valonMin = s.value(QString("valonSynth/minFreq"),500.0).toDouble();
    double valonMax = s.value(QString("valonSynth/maxFreq"),6000.0).toDouble();
    double awgMin = s.value(QString("awg/minFreq"),100.0).toDouble();
    double awgMax = s.value(QString("awg/maxFreq"),6250.0).toDouble();

    d_txSidebandSign = sideband(ui->txSidebandComboBox->currentIndex());
    d_rxSidebandSign = sideband(ui->rxSidebandComboBox->currentIndex());

    ui->chirpLOFrequencyDoubleSpinBox->setMinimum(valonMin*static_cast<double>(ui->valonRxMultiplicationSpinBox->value()));
    ui->chirpLOFrequencyDoubleSpinBox->setMaximum(valonMax*static_cast<double>(ui->valonRxMultiplicationSpinBox->value()));

    if(static_cast<int>(d_currentRxMult) != ui->valonRxMultiplicationSpinBox->value())
    {
        double newFreq = ui->chirpLOFrequencyDoubleSpinBox->value()/d_currentRxMult;
        d_currentRxMult = static_cast<double>(ui->valonRxMultiplicationSpinBox->value());
        newFreq *= d_currentRxMult;
        ui->chirpLOFrequencyDoubleSpinBox->setValue(newFreq);
    }
    else
        ui->chirpLOFrequencyDoubleSpinBox->setValue(d_valonRxFreq*static_cast<double>(ui->valonRxMultiplicationSpinBox->value()));

    ui->valonRxFrequencyDisplayLabel->setText(QString("%1 MHz").arg(d_valonRxFreq,0,'f',3));
    ui->valonTxFrequencyDoubleSpinBox->setValue(d_valonTxFreq);

    double chirpMin = qMin(
                static_cast<double>(ui->totalMultiplicationSpinBox->value())*(d_txSidebandSign*static_cast<double>(ui->awgMultiplicationSpinBox->value())*awgMin + static_cast<double>(ui->valonTxMultiplicationSpinBox->value())*d_valonTxFreq),
                static_cast<double>(ui->totalMultiplicationSpinBox->value())*(d_txSidebandSign*static_cast<double>(ui->awgMultiplicationSpinBox->value())*awgMax + static_cast<double>(ui->valonTxMultiplicationSpinBox->value())*d_valonTxFreq));

    double chirpMax = qMax(
                static_cast<double>(ui->totalMultiplicationSpinBox->value())*(d_txSidebandSign*static_cast<double>(ui->awgMultiplicationSpinBox->value())*awgMin + static_cast<double>(ui->valonTxMultiplicationSpinBox->value())*d_valonTxFreq),
                static_cast<double>(ui->totalMultiplicationSpinBox->value())*(d_txSidebandSign*static_cast<double>(ui->awgMultiplicationSpinBox->value())*awgMax + static_cast<double>(ui->valonTxMultiplicationSpinBox->value())*d_valonTxFreq));

    s.setValue(QString("chirpConfig/chirpMin"),chirpMin);
    s.setValue(QString("chirpConfig/chirpMax"),chirpMax);
    s.sync();

    double bandwidth = s.value(QString("ftmwScope/bandwidth"),16000.0).toDouble();

    double ftMin = qMin(ui->chirpLOFrequencyDoubleSpinBox->value() + d_rxSidebandSign*bandwidth,ui->chirpLOFrequencyDoubleSpinBox->value());
    double ftMax = qMax(ui->chirpLOFrequencyDoubleSpinBox->value() + d_rxSidebandSign*bandwidth,ui->chirpLOFrequencyDoubleSpinBox->value());

    ui->chirpRangeDisplayLabel->setText(QString("%1 – %2 MHz").arg(chirpMin,0,'f',3).arg(chirpMax,0,'f',3));
    ui->ftRangeDisplayLabel->setText(QString("%1 – %2 MHz").arg(ftMin,0,'f',3).arg(ftMax,0,'f',3));

    blockSignals(false);

}

void RfConfigWidget::saveSettings()
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(QString("chirpConfig"));

    s.setValue(QString("awgMult"),ui->awgMultiplicationSpinBox->value());
    s.setValue(QString("txValonMult"),ui->valonTxMultiplicationSpinBox->value());
    s.setValue(QString("txMult"),ui->totalMultiplicationSpinBox->value());
    s.setValue(QString("txSidebandSign"),sideband(ui->txSidebandComboBox->currentIndex()));
    s.setValue(QString("rxValonMult"),ui->valonRxMultiplicationSpinBox->value());
    s.setValue(QString("rxSidebandSign"),sideband(ui->rxSidebandComboBox->currentIndex()));
    s.setValue(QString("loFreq"),ui->chirpLOFrequencyDoubleSpinBox->value());

    s.endGroup();
    s.sync();

}

int RfConfigWidget::sidebandIndex(const double d)
{
    if(d < 0)
        return 1;
    else
        return 0;
}

double RfConfigWidget::sideband(const int index)
{
    if(index == 0)
        return 1.0;
    else
        return -1.0;
}
