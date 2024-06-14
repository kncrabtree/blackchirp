#include "pressurecontrolwidget.h"

#include <hardware/optional/pressurecontroller/pressurecontroller.h>

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QDoubleSpinBox>
#include <QPushButton>
#include <QLabel>
#include <functional>

PressureControlWidget::PressureControlWidget(const PressureControllerConfig &cfg, QWidget *parent) : QWidget(parent), d_config{cfg}
{
    auto vbl = new QVBoxLayout;

    using namespace BC::Key::PController;
    SettingsStorage s(key,SettingsStorage::Hardware);

    QLabel *psLabel = new QLabel("Setpoint");
    psLabel->setAlignment(Qt::AlignRight);

    p_setpointBox = new QDoubleSpinBox(this);
    p_setpointBox->setMinimum(s.get(min,-1.0));
    p_setpointBox->setMaximum(s.get(max,20.0));
    p_setpointBox->setDecimals(s.get(decimals,4));
    p_setpointBox->setSuffix(QString(" ")+s.get(units,QString("")));

    p_setpointBox->setSingleStep(qAbs(p_setpointBox->maximum() - p_setpointBox->minimum())/100.0);
    p_setpointBox->setKeyboardTracking(false);
    connect(p_setpointBox,qOverload<double>(&QDoubleSpinBox::valueChanged),this,[this](double d){
        emit setpointChanged(d_config.headerKey(),d);
    });

    p_controlButton = new QPushButton("Off");
    p_controlButton->setCheckable(true);
    p_controlButton->setChecked(false);
    connect(p_controlButton,&QPushButton::toggled,this,[this](bool b){
        emit pressureControlModeChanged(d_config.headerKey(),b);
    });

    auto hbl = new QHBoxLayout;
    hbl->addWidget(psLabel);
    hbl->addWidget(p_setpointBox,1);
    hbl->addWidget(p_controlButton);

    vbl->addLayout(hbl);

    if(s.get(hasValve,false))
    {
        auto openButton = new QPushButton("Open Valve");
        connect(openButton,&QPushButton::clicked,this,[this](){
            emit valveOpen(d_config.headerKey());
        });

        auto closeButton = new QPushButton("Close Valve");
        connect(closeButton,&QPushButton::clicked,this,[this](){
            emit valveClose(d_config.headerKey());
        });

        auto hbl2 = new QHBoxLayout;
        hbl2->addWidget(openButton,1);
        hbl2->addWidget(closeButton,1);

        vbl->addLayout(hbl2);
    }

    setLayout(vbl);

    pressureSetpointUpdate(cfg.headerKey(),cfg.d_setPoint);
    pressureControlModeUpdate(cfg.headerKey(),cfg.d_pressureControlMode);
}

PressureControllerConfig &PressureControlWidget::toConfig()
{
    d_config.d_pressureControlMode = p_controlButton->isChecked();
    d_config.d_setPoint = p_setpointBox->value();

    return d_config;
}

void PressureControlWidget::pressureSetpointUpdate(const QString key, double p)
{
    if(key != d_config.headerKey())
        return;


    if(!p_setpointBox->hasFocus())
    {
        p_setpointBox->blockSignals(true);
        p_setpointBox->setValue(p);
        p_setpointBox->blockSignals(false);
    }
}

void PressureControlWidget::pressureControlModeUpdate(const QString key, bool en)
{
    if(key != d_config.headerKey())
        return;

    p_controlButton->blockSignals(true);
    if(en)
        p_controlButton->setText("On");
    else
        p_controlButton->setText("Off");
    p_controlButton->setChecked(en);
    p_controlButton->blockSignals(false);

}
