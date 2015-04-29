#include "lifscope.h"

LifScope::LifScope(QObject *parent) :
    HardwareObject(parent)
{
    d_key = QString("lifScope");
}

LifScope::~LifScope()
{

}

