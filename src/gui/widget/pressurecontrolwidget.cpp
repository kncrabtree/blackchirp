#include "pressurecontrolwidget.h"

#include <hardware/optional/pressurecontroller/pressurecontroller.h>

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QDoubleSpinBox>
#include <QPushButton>
#include <QLabel>

PressureControlWidget::PressureControlWidget(QWidget *parent) : QWidget(parent)
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
    connect(p_setpointBox,qOverload<double>(&QDoubleSpinBox::valueChanged),this,&PressureControlWidget::setpointChanged);

    p_controlButton = new QPushButton("Off");
    p_controlButton->setCheckable(true);
    p_controlButton->setChecked(false);
    connect(p_controlButton,&QPushButton::toggled,this,&PressureControlWidget::pressureControlModeChanged);

    auto hbl = new QHBoxLayout;
    hbl->addWidget(psLabel);
    hbl->addWidget(p_setpointBox,1);
    hbl->addWidget(p_controlButton);

    vbl->addLayout(hbl);

    if(s.get(hasValve,false))
    {
        auto openButton = new QPushButton("Open Valve");
        connect(openButton,&QPushButton::clicked,this,&PressureControlWidget::valveOpen);

        auto closeButton = new QPushButton("Close Valve");
        connect(closeButton,&QPushButton::clicked,this,&PressureControlWidget::valveClose);

        auto hbl2 = new QHBoxLayout;
        hbl2->addWidget(openButton,1);
        hbl2->addWidget(closeButton,1);

        vbl->addLayout(hbl2);
    }

    setLayout(vbl);
}

void PressureControlWidget::initialize(const PressureControllerConfig &cfg)
{
    pressureSetpointUpdate(cfg.d_setPoint);
    pressureControlModeUpdate(cfg.d_pressureControlMode);
}

void PressureControlWidget::pressureSetpointUpdate(double p)
{
    if(!p_setpointBox->hasFocus())
    {
        p_setpointBox->blockSignals(true);
        p_setpointBox->setValue(p);
        p_setpointBox->blockSignals(false);
    }
}

void PressureControlWidget::pressureControlModeUpdate(bool en)
{
    p_controlButton->blockSignals(true);
    if(en)
        p_controlButton->setText("On");
    else
        p_controlButton->setText("Off");
    p_controlButton->setChecked(en);
    p_controlButton->blockSignals(false);

}
