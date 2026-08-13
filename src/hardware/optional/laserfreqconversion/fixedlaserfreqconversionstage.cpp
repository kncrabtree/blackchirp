#include <hardware/optional/laserfreqconversion/fixedlaserfreqconversionstage.h>
#include <hardware/core/hardwareregistration.h>

// Register hardware implementation
REGISTER_HARDWARE_META(FixedLaserFreqConversionStage, "Fixed Laser Frequency Conversion Stage")
REGISTER_HARDWARE_PROTOCOLS(FixedLaserFreqConversionStage, CommunicationProtocol::Virtual)

FixedLaserFreqConversionStage::FixedLaserFreqConversionStage(const QString& label, QObject *parent) :
    LaserFreqConversionStage(QString(FixedLaserFreqConversionStage::staticMetaObject.className()), label, parent)
{
}

void FixedLaserFreqConversionStage::initialize()
{
}

bool FixedLaserFreqConversionStage::testConnection()
{
    return true;
}

void FixedLaserFreqConversionStage::setPos(double localCm1)
{
    // No physical actuation: this node stands in for a crystal/compensator
    // not under Blackchirp's control. Recording the commanded value lets
    // readPos() report it back, so the base class's move verification
    // always succeeds.
    d_pos = localCm1;
}

double FixedLaserFreqConversionStage::readPos()
{
    return d_pos;
}
