#include <hardware/optional/laserfreqconversion/virtuallaserfreqconversionstage.h>
#include <hardware/core/hardwareregistration.h>

// Register hardware implementation
REGISTER_HARDWARE_META(VirtualLaserFreqConversionStage, "Virtual Laser Frequency Conversion Stage")
REGISTER_HARDWARE_PROTOCOLS(VirtualLaserFreqConversionStage, CommunicationProtocol::Virtual)

VirtualLaserFreqConversionStage::VirtualLaserFreqConversionStage(const QString& label, QObject *parent) :
    LaserFreqConversionStage(QString(VirtualLaserFreqConversionStage::staticMetaObject.className()), label, parent),
    d_pos(0.0)
{
}

void VirtualLaserFreqConversionStage::initialize()
{
}

bool VirtualLaserFreqConversionStage::testConnection()
{
    d_pos = 10000.0;

    return true;
}

void VirtualLaserFreqConversionStage::setPos(double localCm1)
{
    d_pos = localCm1;
}

double VirtualLaserFreqConversionStage::readPos()
{
    return d_pos;
}
