#include "gascontrolwidget.h"

#include <QGridLayout>
#include <QDoubleSpinBox>
#include <QLineEdit>
#include <QPushButton>
#include <QLabel>

#include <hardware/optional/flowcontroller/flowcontroller.h>


GasControlWidget::GasControlWidget(const FlowConfig &cfg, QWidget *parent) : QWidget(parent), SettingsStorage(BC::Key::GasControl::key.arg(cfg.headerKey()).arg(cfg.hwSubKey())), d_config{cfg}
{
    using namespace BC::Key::GasControl;
    auto gasControlBoxLayout = new QGridLayout;

    gasControlBoxLayout->addWidget(new QLabel("Ch"),0,0,1,1);
    gasControlBoxLayout->addWidget(new QLabel("Gas Name"),0,1,1,1,Qt::AlignCenter);
    gasControlBoxLayout->addWidget(new QLabel("Setpoint"),0,2,1,1);
    gasControlBoxLayout->setColumnStretch(0,0);
    gasControlBoxLayout->setColumnStretch(1,1);
    gasControlBoxLayout->setColumnStretch(2,0);
    gasControlBoxLayout->setMargin(3);
    gasControlBoxLayout->setSpacing(3);


    SettingsStorage fc(cfg.headerKey(),Hardware);
    auto flowChannels = fc.get(BC::Key::Flow::flowChannels,4);
    for(int i=0; i<flowChannels; ++i)
    {
        if((std::size_t) i >= getArraySize(channels))
            appendArrayMap(channels,{});


        auto nameEdit = new QLineEdit;
        nameEdit->setText(getArrayValue(channels,i,gasName,QString("")));
        nameEdit->setSizePolicy(QSizePolicy::MinimumExpanding,QSizePolicy::Preferred);
        connect(nameEdit,&QLineEdit::editingFinished,[this,nameEdit,i](){
            setArrayValue(channels,i,gasName,nameEdit->text(),true);
            emit nameUpdate(d_config.headerKey(),i,nameEdit->text());
        });

        auto controlBox = new QDoubleSpinBox;
        controlBox->setSpecialValueText(QString("Off"));
        controlBox->setKeyboardTracking(false);
        connect(controlBox,static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged),
                [this,i](double v) { emit gasSetpointUpdate(d_config.headerKey(),i,v); } );

        d_widgets.append({nameEdit,controlBox});

        gasControlBoxLayout->addWidget(new QLabel(QString::number(i+1)),1+i,0,1,1,Qt::AlignRight|Qt::AlignVCenter);
        gasControlBoxLayout->addWidget(nameEdit,1+i,1,1,1);
        gasControlBoxLayout->addWidget(controlBox,1+i,2,1,1);
    }


    p_pressureSetpointBox = new QDoubleSpinBox;
    connect(p_pressureSetpointBox,static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged),[this](double sp){
        emit pressureSetpointUpdate(d_config.headerKey(),sp);
    });

    p_pressureControlButton = new QPushButton;
    p_pressureControlButton->setCheckable(true);
    p_pressureControlButton->setChecked(false);
    p_pressureControlButton->setText("Off");
    connect(p_pressureControlButton,&QPushButton::toggled,[this](bool en){
        emit pressureControlUpdate(d_config.headerKey(),en);
        if(en)
            p_pressureControlButton->setText(QString("On"));
        else
            p_pressureControlButton->setText(QString("Off"));
    });

    gasControlBoxLayout->addWidget(new QLabel(QString("Pressure Setpoint")),1+flowChannels,1,1,1,Qt::AlignRight);
    gasControlBoxLayout->addWidget(p_pressureSetpointBox,1+flowChannels,2,1,1);
    gasControlBoxLayout->addWidget(new QLabel(QString("Pressure Control Mode")),2+flowChannels,1,1,1,Qt::AlignRight);
    gasControlBoxLayout->addWidget(p_pressureControlButton,2+flowChannels,2,1,1);

    setLayout(gasControlBoxLayout);
    applySettings();
    initialize(d_config);
}

FlowConfig &GasControlWidget::getFlowConfig()
{

    d_config.d_pressureControlMode = p_pressureControlButton->isChecked();
    d_config.d_pressureSetpoint = p_pressureSetpointBox->value();
    for(int i=0; i<d_widgets.size(); i++)
    {
        auto [name,sp] = d_widgets.at(i);
        d_config.setCh(i,FlowConfig::Setpoint,sp->value());
        d_config.setCh(i,FlowConfig::Name,name->text());
    }

    return d_config;
}

void GasControlWidget::initialize(const FlowConfig &cfg)
{
    if(cfg.headerKey() != d_config.headerKey())
        return;

     for(int i=0; i<cfg.size(); ++i)
         updateGasSetpoint(cfg.headerKey(),i,cfg.setting(i,FlowConfig::Setpoint).toDouble());

     updatePressureSetpoint(cfg.headerKey(),cfg.d_pressureSetpoint);
     updatePressureControl(cfg.headerKey(),cfg.d_pressureControlMode);
}

void GasControlWidget::applySettings()
{
    using namespace BC::Key::Flow;
    SettingsStorage fc(flowController,Hardware);

    p_pressureSetpointBox->setDecimals(fc.get(pDec,3));
    p_pressureSetpointBox->setMaximum(fc.get(pMax,10.0));
    p_pressureSetpointBox->setSuffix(QString(" ") + fc.get(pUnits,QString("")));

    for(int i=0; i<d_widgets.size(); ++i)
    {
        auto b = std::get<QDoubleSpinBox*>(d_widgets.at(i));
        b->setDecimals(fc.getArrayValue(channels,i,chDecimals,2));
        b->setMaximum(fc.getArrayValue(channels,i,chMax,10000.0));
        b->setSuffix(QString(" ")+fc.getArrayValue(channels,i,chUnits,QString("")));

    }
}

void GasControlWidget::updateGasSetpoint(const QString key, int i, double sp)
{
    if(key != d_config.headerKey())
        return;

    if(i < 0 || i >= d_widgets.size())
        return;

    auto b = std::get<QDoubleSpinBox*>(d_widgets.at(i));
    if(!b->hasFocus())
    {
        b->blockSignals(true);
        b->setValue(sp);
        b->blockSignals(false);
    }
}

void GasControlWidget::updatePressureSetpoint(const QString key, double sp)
{
    if(key != d_config.headerKey())
        return;

    if(!p_pressureSetpointBox->hasFocus())
    {
        p_pressureSetpointBox->blockSignals(true);
        p_pressureSetpointBox->setValue(sp);
        p_pressureSetpointBox->blockSignals(false);
    }
}

void GasControlWidget::updatePressureControl(const QString key, bool en)
{
    if(key != d_config.headerKey())
        return;

    if(!p_pressureControlButton->hasFocus())
    {
        blockSignals(true);
        p_pressureControlButton->setChecked(en);
        blockSignals(false);
    }
}

QSize GasControlWidget::sizeHint() const
{
    return {300,400};
}
