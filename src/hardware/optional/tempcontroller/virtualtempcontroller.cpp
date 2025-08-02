#include "virtualtempcontroller.h"
#include <hardware/core/hardwareregistration.h>

#include <QTimer>
#include <QRandomGenerator>

using namespace BC::Key::TC;

// Register hardware implementation
REGISTER_HARDWARE(VirtualTemperatureController, BC::Key::TC::vtcName, "Virtual TemperatureController for Testing")

VirtualTemperatureController::VirtualTemperatureController(QObject *parent) :
    TemperatureController(BC::Key::Comm::hwVirtual,BC::Key::TC::vtcName,
                          CommunicationProtocol::Virtual,4,parent)
{
    save();
}

VirtualTemperatureController::~VirtualTemperatureController()
{
}

bool VirtualTemperatureController::tcTestConnection()
{
    return true;
}
void VirtualTemperatureController::tcInitialize()
{
}

double VirtualTemperatureController::readHwTemperature(const uint ch)
{
    Q_UNUSED(ch)
    auto qr = QRandomGenerator::global();
    return qr->bounded(2.0) + 4.0;
}
