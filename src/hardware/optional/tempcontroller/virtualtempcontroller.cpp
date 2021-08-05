#include "virtualtempcontroller.h"

#include <QTimer>

using namespace BC::Key::TC;

VirtualTemperatureController::VirtualTemperatureController(QObject *parent) :
    TemperatureController(BC::Key::Comm::hwVirtual,BC::Key::vtcName,
                          CommunicationProtocol::Virtual,4,parent)
{
    set(BC::Key::HW::rInterval,10);
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

double VirtualTemperatureController::readHwTemperature(const int ch)
{
    Q_UNUSED(ch)
    return static_cast<double>((qrand() % 65536) - 32768) / 32768.0 + 5.0;
}
