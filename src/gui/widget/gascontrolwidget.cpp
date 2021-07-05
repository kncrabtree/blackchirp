#include "gascontrolwidget.h"

#include <QGridLayout>
#include <QDoubleSpinBox>
#include <QLineEdit>
#include <QPushButton>
#include <QLabel>

#include <hardware/optional/flowcontroller/flowcontroller.h>


GasControlWidget::GasControlWidget(QWidget *parent) : QWidget(parent), SettingsStorage(BC::Key::GasControl::key)
{
    using namespace BC::Key::GasControl;
    auto gasControlBoxLayout = new QGridLayout(this);

    gasControlBoxLayout->addWidget(new QLabel("Ch"),0,0,1,1);
    gasControlBoxLayout->addWidget(new QLabel("Gas Name"),0,1,1,1,Qt::AlignCenter);
    gasControlBoxLayout->addWidget(new QLabel("Setpoint"),0,2,1,1);
    gasControlBoxLayout->setColumnStretch(0,0);
    gasControlBoxLayout->setColumnStretch(1,1);
    gasControlBoxLayout->setColumnStretch(2,0);
    gasControlBoxLayout->setMargin(3);
    gasControlBoxLayout->setSpacing(3);




    SettingsStorage fc(BC::Key::Flow::flowController,Hardware);
    auto flowChannels = fc.get(BC::Key::Flow::flowChannels,4);
    for(int i=0; i<flowChannels; ++i)
    {
        if((std::size_t) i >= getArraySize(channels))
            appendArrayMap(channels,{});


        auto nameEdit = new QLineEdit;
        nameEdit->setText(getArrayValue(channels,i,gasName,QString("")));
        nameEdit->setSizePolicy(QSizePolicy::Ignored,QSizePolicy::Minimum);
        connect(nameEdit,&QLineEdit::editingFinished,[this,nameEdit,i](){
            setArrayValue(channels,i,gasName,nameEdit->text());
            emit nameUpdate(i,nameEdit->text());
        });

        auto controlBox = new QDoubleSpinBox;
        controlBox->setSpecialValueText(QString("Off"));
        controlBox->setKeyboardTracking(false);
        controlBox->setSizePolicy(QSizePolicy::Preferred,QSizePolicy::Minimum);
        connect(controlBox,static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged),
                [this,i](double v) { emit gasSetpointUpdate(i,v); } );

        d_widgets.append({nameEdit,controlBox});

        gasControlBoxLayout->addWidget(new QLabel(QString::number(i+1)),1+i,0,1,1,Qt::AlignRight|Qt::AlignVCenter);
        gasControlBoxLayout->addWidget(nameEdit,1+i,1,1,1);
        gasControlBoxLayout->addWidget(controlBox,1+i,2,1,1);
    }


    p_pressureSetpointBox = new QDoubleSpinBox;
    p_pressureSetpointBox->setSizePolicy(QSizePolicy::Preferred,QSizePolicy::Minimum);
    connect(p_pressureSetpointBox,static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged),this,
            &GasControlWidget::pressureSetpointUpdate);

    p_pressureControlButton = new QPushButton;
    p_pressureControlButton->setCheckable(true);
    p_pressureControlButton->setChecked(false);
    p_pressureControlButton->setText("Off");
    p_pressureControlButton->setSizePolicy(QSizePolicy::Preferred,QSizePolicy::Minimum);
    connect(p_pressureControlButton,&QPushButton::toggled,[this](bool en){
        emit pressureControlUpdate(en);
        if(en)
            p_pressureControlButton->setText(QString("On"));
        else
            p_pressureControlButton->setText(QString("Off"));
    });

    gasControlBoxLayout->addWidget(new QLabel(QString("Pressure Setpoint")),1+flowChannels,1,1,1,Qt::AlignRight);
    gasControlBoxLayout->addWidget(p_pressureSetpointBox,1+flowChannels,2,1,1);
    gasControlBoxLayout->addWidget(new QLabel(QString("Pressure Control Mode")),2+flowChannels,1,1,1,Qt::AlignRight);
    gasControlBoxLayout->addWidget(p_pressureControlButton,2+flowChannels,2,1,1);
    gasControlBoxLayout->addItem(new QSpacerItem(10,10,QSizePolicy::Minimum,QSizePolicy::Expanding),3+flowChannels,0,1,3);

    applySettings();
}

FlowConfig GasControlWidget::getFlowConfig() const
{
    FlowConfig cfg;
    cfg.setPressureControlMode(p_pressureControlButton->isChecked());
    cfg.setPressureSetpoint(p_pressureSetpointBox->value());
    for(int i=0; i<d_widgets.size(); i++)
    {
        auto [name,sp] = d_widgets.at(i);
        cfg.add(sp->value(),name->text());
    }

    return cfg;
}

QStringList GasControlWidget::getGasNames() const
{
    QStringList out;
    for(auto w : d_widgets)
        out << std::get<0>(w)->text();
    return out;
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
        auto b = std::get<1>(d_widgets.at(i));
        b->setDecimals(fc.getArrayValue(channels,i,chDecimals,2));
        b->setMaximum(fc.getArrayValue(channels,i,chMax,10000.0));
        b->setSuffix(QString(" ")+fc.getArrayValue(channels,i,chUnits,QString("")));

    }
}

void GasControlWidget::updateGasSetpoint(int i, double sp)
{
    if(i < 0 || i >= d_widgets.size())
        return;

    auto b = std::get<1>(d_widgets.at(i));
    b->blockSignals(true);
    b->setValue(sp);
    b->blockSignals(false);
}

void GasControlWidget::updatePressureSetpoint(double sp)
{
    p_pressureSetpointBox->blockSignals(true);
    p_pressureSetpointBox->setValue(sp);
    p_pressureSetpointBox->blockSignals(false);
}

void GasControlWidget::updatePressureControl(bool en)
{
    blockSignals(true);
    p_pressureControlButton->setChecked(en);
    blockSignals(false);
}