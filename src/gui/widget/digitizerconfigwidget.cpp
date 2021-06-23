#include <src/gui/widget/digitizerconfigwidget.h>
#include "ui_digitizerconfigwidget.h"

#include <QApplication>
#include <QDoubleValidator>

#include <src/hardware/core/ftmwdigitizer/ftmwscope.h>

DigitizerConfigWidget::DigitizerConfigWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::DigitizerConfigWidget)
{
    ui->setupUi(this);

    ///TODO: Customize more UI settings according to Hardware limits for ftmwscope implementation

    SettingsStorage s(BC::Key::FtmwScope::ftmwScope,SettingsStorage::Hardware);
    auto sr = s.getArray(BC::Key::FtmwScope::sampleRates);
    if(sr.size() > 0)
    {
        for(auto m : sr)
        {
            auto txt = m.find(BC::Key::FtmwScope::srText);
            auto val = m.find(BC::Key::FtmwScope::srValue);
            if(txt != m.end() && val != m.end())
                ui->sampleRateComboBox->addItem(txt->second.toString(),val->second);
        }
    }
    else
    {
        //this code is not tested!
        ui->sampleRateComboBox->setEditable(true);
        auto v = new QDoubleValidator(this);
        v->setRange(0,1e11,0);
        v->setNotation(QDoubleValidator::ScientificNotation);
        ui->sampleRateComboBox->setValidator(v);
    }

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
    ui->triggerSlopeComboBox->setCurrentIndex(ui->triggerSlopeComboBox->findData(qVariantFromValue(sc.slope)));
    int index = ui->sampleRateComboBox->findData(sc.sampleRate);
    if(index < 0)
    {
        if(!ui->sampleRateComboBox->isEditable())
        {
            ui->sampleRateComboBox->setEditable(true);
            auto v = new QDoubleValidator(this);
            v->setRange(0,1e11,0);
            v->setNotation(QDoubleValidator::ScientificNotation);
            ui->sampleRateComboBox->setValidator(v);
        }
    }
    ui->sampleRateComboBox->setCurrentIndex(index);
    ui->recordLengthSpinBox->setValue(sc.recordLength);
    ui->bytesPointSpinBox->setValue(sc.bytesPerPoint);
    ///TODO: Use information from ChirpConfig here

    auto cc = config.chirpConfig();

    SettingsStorage s(BC::Key::FtmwScope::ftmwScope,SettingsStorage::Hardware);
    bool ba = s.get(BC::Key::FtmwScope::blockAverage,false);

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
    SettingsStorage s(BC::Key::FtmwScope::ftmwScope,SettingsStorage::Hardware);
    bool canSf = s.get(BC::Key::FtmwScope::summaryRecord,false);

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

    SettingsStorage s(BC::Key::FtmwScope::ftmwScope,SettingsStorage::Hardware);
    bool canSf = s.get(BC::Key::FtmwScope::summaryRecord,false);
    bool ba = s.get(BC::Key::FtmwScope::blockAverage,false);
    bool ffba = s.get(BC::Key::FtmwScope::multiBlock,false);
    bool ff = ui->fastFrameEnabledCheckBox->isChecked();

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
