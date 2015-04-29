#include "lifscope.h"

LifScope::LifScope(QObject *parent) :
    HardwareObject(parent)
{
    d_key = QString("lifScope");

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    d_refEnabled = s.value(QString("%1/refEnabled").arg(d_key),false).toBool();

}

LifScope::~LifScope()
{

}

