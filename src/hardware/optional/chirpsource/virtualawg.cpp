#include "virtualawg.h"
#include <hardware/core/hardwareregistration.h>

// Register hardware implementation
REGISTER_HARDWARE_META(VirtualAwg, "Virtual AWG for testing and simulation")

VirtualAwg::VirtualAwg(const QString& label, QObject *parent) :
    AWG(QString(VirtualAwg::staticMetaObject.className()), label, parent)
{
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
