#include "liflaserstatusbox.h"

#include <QDoubleSpinBox>
#include <QHBoxLayout>
#include <QLabel>

#include <modules/lif/hardware/liflaser/liflaser.h>
#include <gui/widget/led.h>
#include <data/storage/settingsstorage.h>

LifLaserStatusBox::LifLaserStatusBox(QWidget *parent) : HardwareStatusBox(BC::Key::LifLaser::key,parent)
{
    auto hbl = new QHBoxLayout;

    hbl->addWidget(new QLabel("Position"));

    p_posBox = new QDoubleSpinBox(this);
    p_posBox->setReadOnly(true);
    p_posBox->setKeyboardTracking(false);
    p_posBox->setButtonSymbols(QAbstractSpinBox::NoButtons);
    p_posBox->setFocusPolicy(Qt::ClickFocus);

    hbl->addWidget(p_posBox,1);

    using namespace BC::Key::LifLaser;
    SettingsStorage s(BC::Key::hwKey(key,0),SettingsStorage::Hardware);
    if(s.get(hasFl,true))
    {
        p_led = new Led(this);
        hbl->addWidget(p_led,0);
    }
    else
        p_led = nullptr;

    setLayout(hbl);

    applySettings();
}

void LifLaserStatusBox::applySettings()
{
    using namespace BC::Key::LifLaser;
    SettingsStorage s(BC::Key::hwKey(key,0),SettingsStorage::Hardware);
    p_posBox->setMinimum(s.get(minPos,200.0));
    p_posBox->setMaximum(s.get(maxPos,2000.0));
    p_posBox->setSuffix(QString(" ").append(s.get(units,"nm").toString()));
    p_posBox->setDecimals(s.get(decimals,2));
}

void LifLaserStatusBox::setPosition(double d)
{
    p_posBox->setValue(d);
}

void LifLaserStatusBox::setFlashlampEnabled(bool en)
{
    if(p_led)
        p_led->setState(en);
}
