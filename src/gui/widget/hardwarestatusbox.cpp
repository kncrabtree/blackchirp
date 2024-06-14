#include "hardwarestatusbox.h"

#include <data/storage/settingsstorage.h>
#include <hardware/core/hardwareobject.h>

HardwareStatusBox::HardwareStatusBox(QString key, QWidget *parent) :
    QGroupBox(parent), d_key{key}
{
    SettingsStorage s(d_key,SettingsStorage::Hardware);
    updateTitle(s.get(BC::Key::HW::name,d_key));
    setFlat(true);
}

void HardwareStatusBox::updateTitle(const QString &n)
{
    if(n.size() > 40)
        setTitle(d_titleTemplate.arg(n.mid(0,37)+"..."));
    else
        setTitle(d_titleTemplate.arg(n));
}


QSize HardwareStatusBox::sizeHint() const
{
    return {250,1};
}
