#include "virtualawg.h"

VirtualAwg::VirtualAwg(QObject *parent) :
    AWG(BC::Key::Comm::hwVirtual,BC::Key::vawgName,CommunicationProtocol::Virtual,parent)
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
