#include "pressurestatusbox.h"

#include <QGridLayout>
#include <QDoubleSpinBox>
#include <QLabel>

#include <gui/widget/led.h>
#include <hardware/optional/pressurecontroller/pressurecontroller.h>

PressureStatusBox::PressureStatusBox(QString key, QWidget *parent) : HardwareStatusBox(key,parent)
{
    auto *gl = new QGridLayout;

    gl->addWidget(new QLabel("Chamber"),0,0);

    p_cpBox = new QDoubleSpinBox(this);

    p_cpBox->setReadOnly(true);
    p_cpBox->setButtonSymbols(QAbstractSpinBox::NoButtons);
    p_cpBox->setKeyboardTracking(false);
    p_cpBox->blockSignals(true);

    gl->addWidget(p_cpBox,0,1);

    p_led = new Led(this);
    gl->addWidget(p_led,0,2);

    gl->setColumnStretch(0,0);
    gl->setColumnStretch(1,1);
    gl->setColumnStretch(2,0);

    setLayout(gl);

    updateFromSettings();
}

void PressureStatusBox::pressureUpdate(const QString key, double p)
{
    if(key != d_key)
        return;

    p_cpBox->setValue(p);
}

void PressureStatusBox::pressureControlUpdate(const QString key, bool en)
{
    if(key != d_key)
        return;

    p_led->setState(en);
}

void PressureStatusBox::updateFromSettings()
{
    using namespace BC::Key::PController;
    SettingsStorage s(d_key,SettingsStorage::Hardware);

    p_cpBox->setMinimum(s.get(min,-1.0));
    p_cpBox->setMaximum(s.get(max,20.0));
    p_cpBox->setDecimals(s.get(decimals,4));
    p_cpBox->setSuffix(QString(" ")+s.get(units,QString("")));
}
