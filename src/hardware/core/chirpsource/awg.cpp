#include <src/hardware/core/chirpsource/awg.h>

AWG::AWG(const QString subKey, QObject *parent) : HardwareObject(BC::Key::awg,subKey,parent)
{
}

AWG::~AWG()
{

}
