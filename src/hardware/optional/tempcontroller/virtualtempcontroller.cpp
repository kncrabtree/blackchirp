#include "virtualtempcontroller.h"
#include <hardware/core/hardwareregistration.h>
#include <data/settings/hardwarekeys.h>

#include <QTimer>
#include <QRandomGenerator>

using namespace BC::Key::TC;

// Register hardware implementation
REGISTER_HARDWARE_META(VirtualTemperatureController, "Virtual Temperature Controller for testing and development")

VirtualTemperatureController::VirtualTemperatureController(const QString& label, QObject *parent) :
    TemperatureController(QString(VirtualTemperatureController::staticMetaObject.className()), label, 4, parent)
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
