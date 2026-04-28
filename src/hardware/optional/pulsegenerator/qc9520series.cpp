#include "qcpulsegenerator.h"
#include <hardware/core/hardwareregistration.h>

// Register hardware implementation
REGISTER_HARDWARE_META(Qc9520Series, "QuantumComposers 9520 Series Pulse Generator")
REGISTER_HARDWARE_PROTOCOLS(Qc9520Series,
    CommunicationProtocol::Rs232,
    CommunicationProtocol::Tcp,
    CommunicationProtocol::Gpib)
REGISTER_HARDWARE_SETTINGS(Qc9520Series,
    {BC::Key::PGen::minWidth, "Min Pulse Width (us)",
     "Minimum pulse width in microseconds",
     0.004, 0.0, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::PGen::lockExternal, "External Clock Lock",
     "Default to external 10 MHz clock reference",
     true, QVariant{}, QVariant{}, HwSettingPriority::Optional}
)

Qc9520Series::Qc9520Series(const QString& label, QObject *parent) :
    QCPulseGenerator(QString(Qc9520Series::staticMetaObject.className()), label, 8, parent)
{
}

Qc9520Series::~Qc9520Series()
{

}

void Qc9520Series::initializePGen()
{
    setDefault(BC::Key::Comm::timeout, 200);
    setDefault(BC::Key::Comm::termChar, QString("\r\n"));
}

void Qc9520Series::beginAcquisition()
{
    lockKeys(true);
}

void Qc9520Series::endAcquisition()
{
    lockKeys(false);
}

bool Qc9520Series::pGenWriteCmd(const QString &cmd)
{
    auto resp = pGenQueryCmd(cmd);

    if(resp.startsWith("ok"))
        return true;

    emit hardwareFailure();
    hwError(u"Error writing command %1"_s.arg(cmd));
    return false;
}

QByteArray Qc9520Series::pGenQueryCmd(const QString &cmd)
{
    return p_comm->queryCmd(cmd + "\r\n"_L1);
}
