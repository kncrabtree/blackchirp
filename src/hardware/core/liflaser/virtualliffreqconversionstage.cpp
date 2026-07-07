#include <hardware/core/liflaser/virtualliffreqconversionstage.h>
#include <hardware/core/hardwareregistration.h>

// Register hardware implementation
REGISTER_HARDWARE_META(VirtualLifFreqConversionStage, "Virtual LIF Frequency Conversion Stage")
REGISTER_HARDWARE_PROTOCOLS(VirtualLifFreqConversionStage, CommunicationProtocol::Virtual)

VirtualLifFreqConversionStage::VirtualLifFreqConversionStage(const QString& label, QObject *parent) :
    LifFreqConversionStage(QString(VirtualLifFreqConversionStage::staticMetaObject.className()), label, parent),
    d_pos(0.0)
{
}

void VirtualLifFreqConversionStage::initialize()
{
}

bool VirtualLifFreqConversionStage::testConnection()
{
    d_pos = 10000.0;

    return true;
}

void VirtualLifFreqConversionStage::setPos(double localCm1)
{
    d_pos = localCm1;
}

double VirtualLifFreqConversionStage::readPos()
{
    return d_pos;
}
