#include "virtualtempcontroller.h"

#include <QTimer>

using namespace BC::Key::TC;

VirtualTemperatureController::VirtualTemperatureController(QObject *parent) :
    TemperatureController(BC::Key::hwVirtual,BC::Key::vtcName,
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


QList<double> VirtualTemperatureController::readHWTemperatures()
{
    //not entirely sure what numbers to use here:
    QList<double> out;
    out.reserve(d_temperatureList.size());
    for (int i=0; i<d_temperatureList.size();i++)
        out << static_cast<double>((qrand() % 65536) - 32768) / 32768.0 + 5.0;

    return out;
}

double VirtualTemperatureController::readHwTemperature(const int ch)
{
    Q_UNUSED(ch)
    return static_cast<double>((qrand() % 65536) - 32768) / 32768.0 + 5.0;
}
