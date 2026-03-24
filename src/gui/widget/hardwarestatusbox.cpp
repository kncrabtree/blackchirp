#include "hardwarestatusbox.h"

HardwareStatusBox::HardwareStatusBox(QString key, QWidget *parent) :
    QGroupBox(parent), d_key{key}
{
    auto parts = d_key.split('.');
    if(parts.size() >= 2)
        setTitle(QString("%1: %2").arg(parts[0], parts[1]));
    else
        setTitle(d_key);

    setFlat(true);
}


QSize HardwareStatusBox::sizeHint() const
{
    return {250,1};
}
