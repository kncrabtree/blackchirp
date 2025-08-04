#include "virtualawg.h"
#include <hardware/core/hardwareregistration.h>

// Register hardware implementation
REGISTER_HARDWARE_META(VirtualAwg, "Virtual AWG for testing and simulation")

VirtualAwg::VirtualAwg(const QString& label, QObject *parent) :
    AWG(QString(VirtualAwg::staticMetaObject.className()), label, parent)
{
    setDefault(BC::Key::AWG::rate,16e9);
    setDefault(BC::Key::AWG::samples,2e9);
    setDefault(BC::Key::AWG::min,100.0);
    setDefault(BC::Key::AWG::max,6250);
    setDefault(BC::Key::AWG::prot,true);
    setDefault(BC::Key::AWG::amp,true);
    setDefault(BC::Key::AWG::rampOnly,false);
    setDefault(BC::Key::AWG::triggered,true);
}

VirtualAwg::~VirtualAwg()
{

}

bool VirtualAwg::testConnection()
{
    return true;
}

void VirtualAwg::initialize()
{
}
