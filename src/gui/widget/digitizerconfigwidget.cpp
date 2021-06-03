#include "digitizerconfigwidget.h"
#include "ui_digitizerconfigwidget.h"

#include <QSettings>
#include <QApplication>

DigitizerConfigWidget::DigitizerConfigWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::DigitizerConfigWidget)
{
    ui->setupUi(this);

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

    ui->triggerSlopeComboBox->addItem(QString("Rising Edge"),QVariant::fromValue(BlackChirp::RisingEdge));
    ui->triggerSlopeComboBox->addItem(QString("Falling Edge"),QVariant::fromValue(BlackChirp::FallingEdge));

    ui->triggerDelayDoubleSpinBox->setSuffix(QString::fromUtf16(u" μs"));


    connect(ui->fastFrameEnabledCheckBox,&QCheckBox::toggled,this,&DigitizerConfigWidget::configureUI);
    connect(ui->blockAverageCheckBox,&QCheckBox::toggled,this,&DigitizerConfigWidget::configureUI);

    connect(ui->fIDChannelSpinBox,static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),this,&DigitizerConfigWidget::validateSpinboxes);
    connect(ui->triggerChannelSpinBox,static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),this,&DigitizerConfigWidget::validateSpinboxes);

}

DigitizerConfigWidget::~DigitizerConfigWidget()
{
    delete ui;
}

void DigitizerConfigWidget::setFromConfig(const FtmwConfig config)
{
    d_config = config;
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
    ///TODO: Use information from ChirpConfig here

    auto cc = config.chirpConfig();
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());

    s.beginGroup(QString("ftmwscope"));
    s.beginGroup(s.value(QString("subKey"),QString("virtual")).toString());
    bool ba = s.value(QString("canBlockAverage"),false).toBool();
    s.endGroup();
    s.endGroup();
    ui->fastFrameEnabledCheckBox->blockSignals(true);
    if(cc.numChirps() > 1)
    {
        ui->fastFrameEnabledCheckBox->setEnabled(true);
        ui->framesSpinBox->setEnabled(true);
        ui->summaryFrameCheckBox->setEnabled(true);
        ui->fastFrameEnabledCheckBox->setChecked(true);
        ui->framesSpinBox->setValue(cc.numChirps());
        //Allow user to make stupid settings if they want...
//        ui->fastFrameEnabledCheckBox->setEnabled(false);
//        ui->framesSpinBox->setEnabled(false);
        ui->summaryFrameCheckBox->setChecked(sc.summaryFrame);
    }
    else
    {
        ui->fastFrameEnabledCheckBox->setChecked(false);
        ui->framesSpinBox->setValue(1);
        ui->summaryFrameCheckBox->setChecked(false);
        ui->fastFrameEnabledCheckBox->setEnabled(false);
        ui->framesSpinBox->setEnabled(false);
        ui->summaryFrameCheckBox->setEnabled(false);
    }
    ui->fastFrameEnabledCheckBox->blockSignals(false);
    if(!ba)
    {
        ui->blockAverageCheckBox->setChecked(false);
        ui->blockAverageCheckBox->setEnabled(false);
        ui->averagesSpinBox->setValue(1);
        ui->averagesSpinBox->setEnabled(false);
    }
    else
    {
        ui->blockAverageCheckBox->setChecked(sc.blockAverageEnabled);
        ui->averagesSpinBox->setValue(sc.numAverages);
    }

    configureUI();
    validateSpinboxes();
}

FtmwConfig DigitizerConfigWidget::getConfig()
{
    ///TODO: this should be enforced elsewhere
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());

    s.beginGroup(QString("ftmwscope"));
    s.beginGroup(s.value(QString("subKey"),QString("virtual")).toString());
    bool canSf = s.value(QString("canSummaryFrame"),false).toBool();
    s.endGroup();
    s.endGroup();

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
    if(ui->summaryFrameCheckBox->isChecked())
    {
        sc.summaryFrame = canSf;
        sc.manualFrameAverage = false;
    }
    else
    {
        sc.summaryFrame = false;
        sc.manualFrameAverage = false;
    }
    sc.numFrames = ui->framesSpinBox->value();
    sc.blockAverageEnabled = ui->blockAverageCheckBox->isChecked();
    sc.numAverages = ui->averagesSpinBox->value();
    d_config.setScopeConfig(sc);

    return d_config;
}

void DigitizerConfigWidget::configureUI()
{

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());

    s.beginGroup(QString("ftmwscope"));
    s.beginGroup(s.value(QString("subKey"),QString("virtual")).toString());
    bool ba = s.value(QString("canBlockAverage"),false).toBool();
    bool ffba = s.value(QString("canBlockAndFastFrame"),false).toBool();
    bool canSf = s.value(QString("canSummaryFrame"),false).toBool();
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


    if(ui->fastFrameEnabledCheckBox->isChecked())
    {
        if(canSf)
            ui->summaryFrameCheckBox->setEnabled(ui->fastFrameEnabledCheckBox->isChecked());
        else {
            ui->summaryFrameCheckBox->setChecked(false);
            ui->summaryFrameCheckBox->setEnabled(false);
        }
    }



}

void DigitizerConfigWidget::validateSpinboxes()
{
    blockSignals(true);
    QObject *s = sender();
    if(s == nullptr)
        s = ui->triggerChannelSpinBox;

    QSpinBox *senderBox = dynamic_cast<QSpinBox*>(s);
    if(senderBox == nullptr)
    {
        blockSignals(false);
        return;
    }

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

void DigitizerConfigWidget::setComboBoxIndex(QComboBox *box, QVariant value)
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
