#include "qcpulsegenerator.h"
#include <hardware/core/hardwareregistration.h>

// Register hardware implementation
REGISTER_HARDWARE_META(Qc9510Series, "QuantumComposers 9510 Series Pulse Generator")
REGISTER_HARDWARE_PROTOCOLS(Qc9510Series,
    CommunicationProtocol::Rs232,
    CommunicationProtocol::Tcp,
    CommunicationProtocol::Gpib)
REGISTER_HARDWARE_SETTINGS(Qc9510Series,
    {BC::Key::PGen::numChannels, "Number of Channels",
     "Number of pulse output channels",
     8, 1, 64, HwSettingPriority::Required},
    {BC::Key::PGen::minWidth, "Min Pulse Width (us)",
     "Minimum pulse width in microseconds",
     0.004, 0.0, QVariant{}, HwSettingPriority::Optional},
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

Qc9510Series::Qc9510Series(const QString& label, QObject *parent) :
    QCPulseGenerator(QString(Qc9510Series::staticMetaObject.className()), label, 8, parent)
{
}

Qc9510Series::~Qc9510Series()
{

}

void Qc9510Series::initializePGen()
{
    setDefault(BC::Key::Comm::timeout, 100);
    setDefault(BC::Key::Comm::termChar, QString("\r\n"));
}

bool Qc9510Series::pGenWriteCmd(QString cmd)
{
    int maxAttempts = 10;
    for(int i=0; i<maxAttempts; i++)
    {
        QByteArray resp = pGenQueryCmd(cmd);
        if(resp.isEmpty())
            break;

        if(resp.startsWith("ok"))
            return true;
    }

    emit hardwareFailure();
    emit logMessage(QString("Error writing command %1").arg(cmd),LogHandler::Error);
    return false;
}

QByteArray Qc9510Series::pGenQueryCmd(QString cmd)
{
    return p_comm->queryCmd(cmd.append(QString("\n")));
}

void Qc9510Series::beginAcquisition()
{
    lockKeys(true);
}

void Qc9510Series::endAcquisition()
{
    lockKeys(false);
}
