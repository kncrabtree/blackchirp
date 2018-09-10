#include "lifcontrolwidget.h"
#include "ui_lifcontrolwidget.h"

#include <QSettings>
#include <QApplication>
#include <math.h>

LifControlWidget::LifControlWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::LifControlWidget)
{
    ui->setupUi(this);

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());

    updateHardwareLimits();

    ui->lIFVScaleDoubleSpinBox->setValue(s.value(QString("lifConfig/lifVScale"),0.02).toDouble());
    ui->samplesSpinBox->setValue(s.value(QString("lifConfig/samples"),1000).toInt());
    ui->sampleRateSpinBox->setValue(static_cast<int>(round(s.value(QString("lifConfig/sampleRate"),1e9).toDouble()/1e6)));
    ui->refEnabledCheckBox->setChecked(s.value(QString("lifConfig/refEnabled"),false).toBool());
    ui->refVScaleDoubleSpinBox->setValue(s.value(QString("lifConfig/refVScale"),0.02).toDouble());
    ui->refVScaleDoubleSpinBox->setEnabled(ui->refEnabledCheckBox->isChecked());


    //connect signals
    auto sig = [=](){ emit updateScope(toConfig()); };
    auto dvc = static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged);
    auto ivc = static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged);
    connect(ui->lIFVScaleDoubleSpinBox,dvc,sig);
    connect(ui->sampleRateSpinBox,ivc,sig);
    connect(ui->samplesSpinBox,ivc,sig);
    connect(ui->refEnabledCheckBox,&QCheckBox::toggled,sig);
    connect(ui->refVScaleDoubleSpinBox,dvc,sig);
    connect(ui->lifPlot,&LifTracePlot::colorChanged,this,&LifControlWidget::lifColorChanged);

    connect(this,&LifControlWidget::newTrace,ui->lifPlot,&LifTracePlot::newTrace);

    connect(ui->refEnabledCheckBox,&QCheckBox::toggled,ui->refVScaleDoubleSpinBox,&QDoubleSpinBox::setEnabled);

    ui->lifPlot->setAxisAutoScaleRange(QwtPlot::xBottom,0.0,static_cast<double>(ui->samplesSpinBox->value())/static_cast<double>(ui->sampleRateSpinBox->value())*1e3);
    ui->lifPlot->autoScale();

}

LifControlWidget::~LifControlWidget()
{
    delete ui;
}

LifConfig LifControlWidget::getSettings(LifConfig c)
{
    c = ui->lifPlot->getSettings(c);
    c.setScopeConfig(toConfig());
    return c;
}

void LifControlWidget::scopeConfigChanged(const BlackChirp::LifScopeConfig c)
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());

    ui->lIFVScaleDoubleSpinBox->blockSignals(true);
    ui->lIFVScaleDoubleSpinBox->setValue(c.vScale1);
    ui->lIFVScaleDoubleSpinBox->blockSignals(false);

    ui->sampleRateSpinBox->blockSignals(true);
    ui->sampleRateSpinBox->setValue(static_cast<int>(round(c.sampleRate/1e6)));
    ui->sampleRateSpinBox->blockSignals(false);

    ui->samplesSpinBox->blockSignals(true);
    ui->samplesSpinBox->setValue(c.recordLength);
    ui->samplesSpinBox->blockSignals(false);

    ui->refEnabledCheckBox->blockSignals(true);
    ui->refEnabledCheckBox->setChecked(c.refEnabled);
    ui->refEnabledCheckBox->blockSignals(false);

    ui->refVScaleDoubleSpinBox->blockSignals(true);
    ui->refVScaleDoubleSpinBox->setValue(c.vScale2);
    ui->refVScaleDoubleSpinBox->blockSignals(false);

    ui->refVScaleDoubleSpinBox->setEnabled(ui->refEnabledCheckBox->isChecked());

    s.setValue(QString("lifConfig/lifVScale"),c.vScale1);
    s.setValue(QString("lifConfig/sampleRate"),c.sampleRate);
    s.setValue(QString("lifConfig/samples"),c.recordLength);
    s.setValue(QString("lifConfig/refEnabled"),c.refEnabled);
    s.setValue(QString("lifConfig/refVScale"),c.vScale2);
    s.sync();

    ui->lifPlot->setAxisAutoScaleRange(QwtPlot::xBottom,0.0,static_cast<double>(c.recordLength)/c.sampleRate*1e9);
    ui->lifPlot->reset();

}

void LifControlWidget::checkLifColors()
{
    ui->lifPlot->checkColors();
}

void LifControlWidget::updateHardwareLimits()
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(QString("lifScope"));
    s.beginGroup(s.value(QString("subKey"),QString("virtual")).toString());
    double minVS = s.value(QString("minVScale"),0.01).toDouble();
    double maxVS = s.value(QString("maxVScale"),5.0).toDouble();
    double minSamples = s.value(QString("minSamples"),1000).toInt();
    double maxSamples = s.value(QString("maxSamples"),10000).toInt();
    s.endGroup();
    s.endGroup();

    ui->lIFVScaleDoubleSpinBox->blockSignals(true);
    ui->lIFVScaleDoubleSpinBox->setRange(minVS,maxVS);
    ui->lIFVScaleDoubleSpinBox->blockSignals(false);

    ui->samplesSpinBox->blockSignals(true);
    ui->samplesSpinBox->setRange(minSamples,maxSamples);
    ui->samplesSpinBox->blockSignals(false);

    ui->refVScaleDoubleSpinBox->blockSignals(true);
    ui->refVScaleDoubleSpinBox->setRange(minVS,maxVS);
    ui->refVScaleDoubleSpinBox->blockSignals(false);
}

BlackChirp::LifScopeConfig LifControlWidget::toConfig() const
{
    BlackChirp::LifScopeConfig out;
    out.vScale1 = ui->lIFVScaleDoubleSpinBox->value();
    out.sampleRate = static_cast<double>(ui->sampleRateSpinBox->value())*1e6;
    out.recordLength = ui->samplesSpinBox->value();
    out.refEnabled = ui->refEnabledCheckBox->isChecked();
    out.vScale2 = ui->refVScaleDoubleSpinBox->value();
    return out;
}
