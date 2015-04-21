#include "ftmwscope.h"
#include <QFile>
#include <QDebug>
#include <math.h>
#include <QTimer>

FtmwScope::FtmwScope(QObject *parent) : HardwareObject(parent)
{
    d_key = QString("ftmwscope");
}



