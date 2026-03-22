#include "hardwarestatusbox.h"
#include <data/settings/hardwarekeys.h>

#include <data/storage/settingsstorage.h>
#include <hardware/core/hardwareobject.h>

HardwareStatusBox::HardwareStatusBox(QString key, QWidget *parent) :
    QGroupBox(parent), d_key{key}
{
    auto parts = d_key.split('.');
    if(parts.size() >= 2)
        setTitle(QString("%1: %2").arg(parts[0], parts[1]));
    else
        setTitle(d_key);

    SettingsStorage s(d_key,SettingsStorage::Hardware);
    updateTitle(s.get(BC::Key::HW::name,d_key));
    setFlat(true);
}

void HardwareStatusBox::updateTitle(const QString &n)
{
    setToolTip(n);
}


QSize HardwareStatusBox::sizeHint() const
{
    return {250,1};
}
