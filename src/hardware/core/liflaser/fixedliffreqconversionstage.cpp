#include <hardware/core/liflaser/fixedliffreqconversionstage.h>
#include <hardware/core/hardwareregistration.h>

// Register hardware implementation
REGISTER_HARDWARE_META(FixedLifFreqConversionStage, "Fixed LIF Frequency Conversion Stage")
REGISTER_HARDWARE_PROTOCOLS(FixedLifFreqConversionStage, CommunicationProtocol::Virtual)

FixedLifFreqConversionStage::FixedLifFreqConversionStage(const QString& label, QObject *parent) :
    LifFreqConversionStage(QString(FixedLifFreqConversionStage::staticMetaObject.className()), label, parent)
{
}

void FixedLifFreqConversionStage::initialize()
{
}

bool FixedLifFreqConversionStage::testConnection()
{
    return true;
}

void FixedLifFreqConversionStage::setPos(double localCm1)
{
    // No physical actuation: this node stands in for a crystal/compensator
    // not under Blackchirp's control. Recording the commanded value lets
    // readPos() report it back, so the base class's move verification
    // always succeeds.
    d_pos = localCm1;
}

double FixedLifFreqConversionStage::readPos()
{
    return d_pos;
}
