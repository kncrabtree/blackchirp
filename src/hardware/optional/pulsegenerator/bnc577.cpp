#include "bnc577.h"
#include <hardware/core/hardwareregistration.h>

using namespace BC::Key::PGen;

// Register hardware implementation
REGISTER_HARDWARE_META(Bnc577, "BNC 577 Pulse Generator")
REGISTER_HARDWARE_PROTOCOLS(Bnc577,
    CommunicationProtocol::Rs232,
    CommunicationProtocol::Tcp,
    CommunicationProtocol::Gpib)
REGISTER_HARDWARE_SETTINGS(Bnc577,
    {BC::Key::PGen::numChannels, "Number of Channels",
     "Number of pulse output channels",
     8, 1, 64, HwSettingPriority::Required},
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

Bnc577::Bnc577(const QString& label, QObject *parent)
    : QCPulseGenerator{QString(Bnc577::staticMetaObject.className()), label, 8, parent}
{
    // Communication defaults
    setDefault(BC::Key::Comm::timeout, 200);
    setDefault(BC::Key::Comm::termChar, QString("\r\n"));
}

void Bnc577::initializePGen()
{
}

bool Bnc577::pGenWriteCmd(QString cmd)
{
    auto resp = pGenQueryCmd(cmd);

    if(resp.startsWith("ok"))
        return true;

    emit hardwareFailure();
    hwError(u"Error writing command %1"_s.arg(cmd));
    return false;
}

QByteArray Bnc577::pGenQueryCmd(QString cmd)
{
    return p_comm->queryCmd(cmd.append(QString("\r\n")));
}
