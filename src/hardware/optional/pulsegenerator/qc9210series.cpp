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
     4, 1, 64, HwSettingPriority::Required},
    {BC::Key::PGen::minWidth, "Min Pulse Width (us)",
     "Minimum pulse width in microseconds",
     0.01, 0.0, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::PGen::maxWidth, "Max Pulse Width (us)",
     "Maximum pulse width in microseconds",
     1e5, 0.0, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::PGen::minDelay, "Min Delay (us)",
     "Minimum channel delay in microseconds",
     0.0, 0.0, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::PGen::maxDelay, "Max Delay (us)",
     "Maximum channel delay in microseconds",
     1e5, 0.0, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::PGen::minRepRate, "Min Rep Rate (Hz)",
     "Minimum repetition rate in Hz",
     0.01, 0.0, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::PGen::maxRepRate, "Max Rep Rate (Hz)",
     "Maximum repetition rate in Hz",
     1e5, 0.0, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::PGen::lockExternal, "External Clock Lock",
     "Default to external 10 MHz clock reference",
     false, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::PGen::canDutyCycle, "Duty Cycle Mode",
     "Supports duty-cycle triggered mode",
     true, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::PGen::canTrigger, "External Trigger",
     "Supports external trigger input",
     true, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::PGen::dutyMax, "Max Duty Cycles",
     "Maximum number of pulses per duty cycle burst",
     100000, 1, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::PGen::canSyncToChannel, "Sync to Channel",
     "Channels can be synchronized to another channel output",
     true, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::PGen::canDisableChannels, "Can Disable Channels",
     "Individual channels can be independently enabled/disabled",
     true, QVariant{}, QVariant{}, HwSettingPriority::Optional}
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
