#include "virtualtempcontroller.h"

#include <QTimer>
#include <QRandomGenerator>

using namespace BC::Key::TC;

VirtualTemperatureController::VirtualTemperatureController(QObject *parent) :
    TemperatureController(BC::Key::Comm::hwVirtual,BC::Key::vtcName,
                          CommunicationProtocol::Virtual,4,parent)
{
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
    auto qr = QRandomGenerator::global();
    return qr->bounded(2.0) + 4.0;
}
