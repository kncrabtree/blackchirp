#include "hardwareobject.h"

HardwareObject::HardwareObject(QString key, QString name, QObject *parent) :
	QObject(parent), d_prettyName(name), d_key(key), d_useTermChar(false), d_timeOut(1000)
{
#ifdef BC_NOHARDWARE
    d_virtual = true;
#else
	d_virtual = false;
#endif
}


void HardwareObject::sleep(bool b)
{
	if(b)
		emit logMessage(name().append(QString(" is asleep.")));
	else
		emit logMessage(name().append(QString(" is awake.")));
}
