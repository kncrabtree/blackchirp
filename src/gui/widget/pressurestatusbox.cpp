#include "pressurestatusbox.h"

#include <QGridLayout>
#include <QLabel>

#include <gui/util/numericformat.h>
#include <gui/widget/led.h>
#include <hardware/optional/pressurecontroller/pressurecontroller.h>

using namespace Qt::Literals::StringLiterals;

PressureStatusBox::PressureStatusBox(const QString &key, QWidget *parent) : HardwareStatusBox(key,parent)
{
    auto *gl = new QGridLayout;

    gl->addWidget(new QLabel("Chamber"),0,0);

    p_cpLabel = new QLabel(this);
    p_cpLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);

    gl->addWidget(p_cpLabel,0,1);

    p_led = new Led(this);
    gl->addWidget(p_led,0,2);

    gl->setColumnStretch(0,0);
    gl->setColumnStretch(1,1);
    gl->setColumnStretch(2,0);

    body()->setLayout(gl);

    updateFromSettings();
}

void PressureStatusBox::pressureUpdate(const QString key, double p)
{
    if(key != d_key)
        return;

    p_cpLabel->setText(BC::Gui::formatNumberForDisplay(p, d_decimals) + d_suffix);
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

    d_decimals = s.get(decimals,4);
    d_suffix = " "_L1 + s.get(units,QString(""));
}
