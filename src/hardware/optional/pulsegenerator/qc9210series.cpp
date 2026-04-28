#include "qcpulsegenerator.h"
#include <hardware/core/hardwareregistration.h>

// Register hardware implementation
REGISTER_HARDWARE_META(Qc9210Series, "QuantumComposers 9210 Series Pulse Generator")
REGISTER_HARDWARE_PROTOCOLS(Qc9210Series,
    CommunicationProtocol::Rs232,
    CommunicationProtocol::Tcp,
    CommunicationProtocol::Gpib)
REGISTER_HARDWARE_SETTINGS(Qc9210Series,
    {BC::Key::PGen::numChannels, "Number of Channels",
     "Number of pulse output channels",
     4, 1, 64, HwSettingPriority::Required}
)

Qc9210Series::Qc9210Series(const QString& label, QObject *parent) :
    QCPulseGenerator(QString(Qc9210Series::staticMetaObject.className()), label, 4, parent)
{
}

Qc9210Series::~Qc9210Series()
{

}

void Qc9210Series::initializePGen()
{
    setDefault(BC::Key::Comm::timeout, 200);
    setDefault(BC::Key::Comm::termChar, QString("\r\n"));
}

void Qc9210Series::beginAcquisition()
{
}

void Qc9210Series::endAcquisition()
{
}

bool Qc9210Series::pGenWriteCmd(const QString &cmd)
{
    auto resp = pGenQueryCmd(cmd);

    if(resp.startsWith("ok"))
        return true;

    emit hardwareFailure();
    hwError(u"Error writing command %1"_s.arg(cmd));
    return false;
}

QByteArray Qc9210Series::pGenQueryCmd(const QString &cmd)
{
    return p_comm->queryCmd(cmd + "\r\n"_L1);
}
