#include <src/hardware/core/ftmwdigitizer/ftmwscope.h>

FtmwScope::FtmwScope(QObject *parent) : HardwareObject(parent)
{
    d_key = QString("ftmwscope");
}

FtmwScope::~FtmwScope()
{

}



