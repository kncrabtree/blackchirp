#include <src/modules/lif/gui/lifcontrolwidget.h>
#include "ui_lifcontrolwidget.h"

#include <QSettings>
#include <QApplication>
#include <cmath>

LifControlWidget::LifControlWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::LifControlWidget)
{
    ui->setupUi(this);

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());

    updateHardwareLimits();

    ui->lIFVScaleDoubleSpinBox->setValue(s.value(QString("lifConfig/lifVScale"),0.02).toDouble());
    ui->samplesSpinBox->setValue(s.value(QString("lifConfig/samples"),1000).toInt());
    setSampleRateBox(s.value(QString("lifConfig/sampleRate"),1e9).toDouble());
    ui->refEnabledCheckBox->setChecked(s.value(QString("lifConfig/refEnabled"),false).toBool());
    ui->refVScaleDoubleSpinBox->setValue(s.value(QString("lifConfig/refVScale"),0.02).toDouble());
    ui->refVScaleDoubleSpinBox->setEnabled(ui->refEnabledCheckBox->isChecked());


    //connect signals
    auto sig = [=](){ emit updateScope(toConfig()); };
    auto dvc = static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged);
    auto ivc = static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged);
    auto cvc = static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged);
    connect(ui->lIFVScaleDoubleSpinBox,dvc,sig);
    connect(ui->sampleRateComboBox,cvc,sig);
    connect(ui->samplesSpinBox,ivc,sig);
    connect(ui->refEnabledCheckBox,&QCheckBox::toggled,sig);
    connect(ui->refVScaleDoubleSpinBox,dvc,sig);
    connect(ui->laserPosDoubleSpinBox,dvc,this,&LifControlWidget::laserPosUpdate);
    connect(this,&LifControlWidget::newTrace,ui->lifPlot,&LifTracePlot::newTrace);

    connect(ui->refEnabledCheckBox,&QCheckBox::toggled,ui->refVScaleDoubleSpinBox,&QDoubleSpinBox::setEnabled);

    ui->lifPlot->setAxisAutoScaleRange(QwtPlot::xBottom,0.0,static_cast<double>(1.0/ui->sampleRateComboBox->currentData().toDouble()*1e9*ui->samplesSpinBox->value()));
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

double LifControlWidget::laserPos() const
{
    return ui->laserPosDoubleSpinBox->value();
}

void LifControlWidget::scopeConfigChanged(const BlackChirp::LifScopeConfig c)
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());

    ui->lIFVScaleDoubleSpinBox->blockSignals(true);
    ui->lIFVScaleDoubleSpinBox->setValue(c.vScale1);
    ui->lIFVScaleDoubleSpinBox->blockSignals(false);

    ui->sampleRateComboBox->blockSignals(true);
    setSampleRateBox(c.sampleRate);
    ui->sampleRateComboBox->blockSignals(false);

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
    int minSamples = s.value(QString("minSamples"),1000).toInt();
    int maxSamples = s.value(QString("maxSamples"),10000).toInt();
    ui->sampleRateComboBox->blockSignals(true);
    bool ok = false;
    double oldrate = ui->sampleRateComboBox->itemData(ui->sampleRateComboBox->currentIndex()).toDouble(&ok);
    int size = s.beginReadArray(QString("sampleRates"));
    ui->sampleRateComboBox->clear();
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
        ui->sampleRateComboBox->addItem(QString("1250 MS/s"),1250e6);
        ui->sampleRateComboBox->addItem(QString("625 MS/s"),625e6);
        ui->sampleRateComboBox->addItem(QString("312.5 MS/s"),312.5e6);
        ui->sampleRateComboBox->addItem(QString("156.25 MS/s"),156.25e6);
        ui->sampleRateComboBox->addItem(QString("78.125 MS/s"),78.125e6);
        ui->sampleRateComboBox->addItem(QString("39.0625 MS/s"),39.0625e6);
    }
    if(ok)
        setSampleRateBox(oldrate);
    s.endArray();
    s.endGroup();
    s.endGroup();
    ui->sampleRateComboBox->blockSignals(false);

    ui->lIFVScaleDoubleSpinBox->blockSignals(true);
    ui->lIFVScaleDoubleSpinBox->setRange(minVS,maxVS);
    ui->lIFVScaleDoubleSpinBox->blockSignals(false);

    ui->samplesSpinBox->blockSignals(true);
    ui->samplesSpinBox->setRange(minSamples,maxSamples);
    ui->samplesSpinBox->blockSignals(false);

    ui->refVScaleDoubleSpinBox->blockSignals(true);
    ui->refVScaleDoubleSpinBox->setRange(minVS,maxVS);
    ui->refVScaleDoubleSpinBox->blockSignals(false);

    ui->laserPosDoubleSpinBox->configure();
}

void LifControlWidget::setLaserPos(double pos)
{
    ui->laserPosDoubleSpinBox->blockSignals(true);
    ui->laserPosDoubleSpinBox->setValue(pos);
    ui->laserPosDoubleSpinBox->blockSignals(false);
}

void LifControlWidget::setSampleRateBox(double rate)
{
    int closest = 0;
    double diff = qAbs(ui->sampleRateComboBox->itemData(0).toDouble()-rate);
    for(int i=1; i<ui->sampleRateComboBox->count(); i++)
    {
        double thisDiff = qAbs(ui->sampleRateComboBox->itemData(i).toDouble()-rate);
        if(thisDiff<diff)
        {
            closest = i;
            diff = thisDiff;
        }
    }
    ui->sampleRateComboBox->setCurrentIndex(closest);
}

BlackChirp::LifScopeConfig LifControlWidget::toConfig() const
{
    BlackChirp::LifScopeConfig out;
    out.vScale1 = ui->lIFVScaleDoubleSpinBox->value();
    out.sampleRate = static_cast<double>(ui->sampleRateComboBox->currentData().toDouble());
    out.recordLength = ui->samplesSpinBox->value();
    out.refEnabled = ui->refEnabledCheckBox->isChecked();
    out.vScale2 = ui->refVScaleDoubleSpinBox->value();
    return out;
}
