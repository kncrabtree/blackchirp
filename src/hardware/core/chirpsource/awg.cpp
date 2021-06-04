#include <src/hardware/core/chirpsource/awg.h>

AWG::AWG(QObject *parent) : HardwareObject(parent)
{
    d_key = QString("awg");
}

AWG::~AWG()
{

}
