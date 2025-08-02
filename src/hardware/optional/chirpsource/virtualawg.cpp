#include "virtualawg.h"
#include <hardware/core/hardwareregistration.h>

// Register hardware implementation
REGISTER_HARDWARE(VirtualAwg, BC::Key::AWG::virtualAwgName, "Virtual AWG for testing and simulation")

VirtualAwg::VirtualAwg(QObject *parent) :
    AWG(BC::Key::Comm::hwVirtual,BC::Key::AWG::virtualAwgName,CommunicationProtocol::Virtual,parent)
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
