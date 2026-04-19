#include "liflaserstatusbox.h"

#include <QHBoxLayout>
#include <QLabel>

#include <hardware/core/liflaser/liflaser.h>
#include <gui/widget/led.h>
#include <gui/util/numericformat.h>
#include <data/storage/settingsstorage.h>

using namespace Qt::Literals::StringLiterals;

LifLaserStatusBox::LifLaserStatusBox(const QString &key, QWidget *parent) : HardwareStatusBox(key,parent)
{
    auto hbl = new QHBoxLayout;

    hbl->addWidget(new QLabel("Position"_L1));

    p_posLabel = new QLabel(this);
    p_posLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    hbl->addWidget(p_posLabel,1);

    using namespace BC::Key::LifLaser;
    SettingsStorage s(d_key,SettingsStorage::Hardware);
    if(s.get(hasFl,true))
    {
        p_led = new Led(this);
        hbl->addWidget(p_led,0);
    }
    else
        p_led = nullptr;

    body()->setLayout(hbl);

    applySettings();
}

void LifLaserStatusBox::applySettings()
{
    using namespace BC::Key::LifLaser;
    SettingsStorage s(d_key,SettingsStorage::Hardware);
    d_decimals = s.get(decimals,2);
    d_suffix = " "_L1 + s.get(units,"nm"_L1).toString();
    p_posLabel->setText(BC::Gui::formatNumberForDisplay(d_position, d_decimals) + d_suffix);
}

void LifLaserStatusBox::setPosition(double d)
{
    d_position = d;
    p_posLabel->setText(BC::Gui::formatNumberForDisplay(d_position, d_decimals) + d_suffix);
}

void LifLaserStatusBox::setFlashlampEnabled(bool en)
{
    if(p_led)
        p_led->setState(en);
}
