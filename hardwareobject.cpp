#include "hardwareobject.h"

HardwareObject::HardwareObject(QString key, QString name, QObject *parent) :
	QObject(parent), d_prettyName(name), d_key(key), d_useTermChar(false), d_timeOut(1000)
{
}


void HardwareObject::sleep(bool b)
{
	if(b)
		emit logMessage(name().append(QString(" is asleep.")));
	else
		emit logMessage(name().append(QString(" is awake.")));
}
