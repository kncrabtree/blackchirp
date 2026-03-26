#ifndef HARDWAREKEYS_H
#define HARDWAREKEYS_H

#include <QString>

//! \file hardwarekeys.h
//! \brief Hardware settings keys that are used across multiple modules
//!
//! This header contains only the hardware keys that are:
//! 1. Defined in base hardware classes (not implementation-specific)
//! 2. Used by UI, experiment logic, or other non-hardware modules
//!
//! Implementation-specific keys should remain in their respective hardware headers.

namespace BC::Key {

// Hardware base class keys (from HardwareObject)
namespace HW {
    static const QString key{"key"}; /*!< Hardware type key */
    static const QString model{"model"}; /*!< Hardware implementation/model name */
    static const QString name{"prettyName"}; /*!< Display name */
    static const QString connected{"connected"}; /*!< Whether last communication was successful */
    static const QString critical{"critical"}; /*!< Whether communication failure should abort an experiment */
static const QString commType{"commType"}; /*!< CommunicationProtocol type */
    static const QString supportedProtocols{"supportedProtocols"}; /*!< List of supported communication protocols */
    static const QString rInterval{"rollingDataIntervalSec"}; /*!< Timer interval for rolling data (seconds) */
}

// Communication protocol keys (from CommunicationProtocol)
namespace Comm {
    static const QString rs232{"rs232"}; /*!< Rs232Instrument */
    static const QString tcp{"tcp"}; /*!< TcpInstrument */
    static const QString gpib{"gpib"}; /*!< GpibInstrument */
    static const QString custom{"custom"}; /*!< CustomInstrument */
    static const QString hwVirtual{"virtual"}; /*!< VirtualInstrument */
    static const QString timeout{"timeout"}; /*!< Read timeout in ms (<=0 disables timeout) */
    static const QString termChar{"termChar"}; /*!< Termination character(s) (empty disables termChar) */
}

// RS232 protocol specific keys (from Rs232Instrument)
namespace RS232 {
    static const QString baud{"baudrate"};
    static const QString id{"id"};
    static const QString dataBits{"databits"};
    static const QString parity{"parity"};
    static const QString stopBits{"stopbits"};
    static const QString flowControl{"flowControl"};
}

// TCP protocol specific keys (from TcpInstrument)
namespace TCP {
    static const QString ip{"ip"};
    static const QString port{"port"};
}

// GPIB protocol specific keys (from GpibInstrument)
namespace GPIB {
    static const QString gpibAddress{"address"};
    static const QString gpibController{"controller"}; // Future support for multiple controllers
}

// Custom protocol specific keys (from CustomInstrument)
namespace Custom {
    static const QString comm{"comm"}; /*!< Key for communication array. */
    static const QString type{"type"}; /*!< Type of data entry field (stringKey or intKey). */
    static const QString key{"key"}; /*!< Identifier key for this data field. */
    static const QString intKey{"int"}; /*!< Key for integer data entry type. */
    static const QString intMin{"min"}; /*!< Minimum allowed integer value. */
    static const QString intMax{"max"}; /*!< Maximum allowed integer value. */
    static const QString stringKey{"string"}; /*!< Key for string data entry type. */
    static const QString maxLen{"length"}; /*!< Maximum allowed length of string. */
    static const QString label{"name"}; /*!< Label displayed to user for this data field. */
}

// FTMW digitizer keys (from FtmwScope)
namespace FtmwScope {
    static const QString ftmwScope{"FtmwDigitizer"}; /*!< FTMW digitizer hardware type key */
    static const QString bandwidth{"bandwidthMHz"}; /*!< Analog bandwidth in MHz */
    static const QString fidCh{"fidChannel"}; /*!< FID recording channel */
}

// Digitizer configuration keys (from DigitizerConfig)
namespace Digi {
    static const QString dwAnChannels{"channels"}; /*!< Analog channels configuration array */
    static const QString dwDigChannels{"digitalChannels"}; /*!< Digital channels configuration array */
    static const QString numAnalogChannels{"numAnalogChannels"}; /*!< Number of analog channels */
    static const QString hasAuxTriggerChannel{"hasAuxTriggerChannel"}; /*!< Whether auxiliary trigger channel exists */
    static const QString numDigitalChannels{"numDigitalChannels"}; /*!< Number of digital channels */
    static const QString minFullScale{"minFullScale"}; /*!< Minimum full scale voltage */
    static const QString maxFullScale{"maxFullScale"}; /*!< Maximum full scale voltage */
    static const QString minVOffset{"minVOffset"}; /*!< Minimum vertical offset */
    static const QString maxVOffset{"maxVOffset"}; /*!< Maximum vertical offset */
    static const QString isTriggered{"isTriggered"}; /*!< Whether digitizer supports triggering */
    static const QString minTrigDelay{"minTrigDelayUs"}; /*!< Minimum trigger delay in microseconds */
    static const QString maxTrigDelay{"maxTrigDelayUs"}; /*!< Maximum trigger delay in microseconds */
    static const QString minTrigLevel{"minTrigLevel"}; /*!< Minimum trigger level */
    static const QString maxTrigLevel{"maxTrigLevel"}; /*!< Maximum trigger level */
    static const QString maxRecordLength{"maxRecordLength"}; /*!< Maximum record length */
    static const QString canMultiRecord{"canMultiRecord"}; /*!< Whether multi-record mode is supported */
    static const QString maxRecords{"maxRecords"}; /*!< Maximum number of records */
    static const QString canBlockAverage{"canBlockAverage"}; /*!< Whether block averaging is supported */
    static const QString maxAverages{"maxAverages"}; /*!< Maximum number of averages */
    static const QString multiBlock{"canBlockAndMultiRecord"}; /*!< Whether block and multi-record modes can be combined */
    static const QString maxBytes{"maxBytesPerPoint"}; /*!< Maximum bytes per data point */
    static const QString sampleRates{"sampleRates"}; /*!< Supported sample rates array */
    static const QString srText{"text"}; /*!< Sample rate display text */
    static const QString srValue{"val"}; /*!< Sample rate numeric value */
}

// Clock configuration keys (from Clock base class)
namespace Clock {
    static const QString clock{"Clock"}; /*!< Clock hardware type key */
    static const QString minFreq{"minFreqMHz"}; /*!< Minimum frequency in MHz */
    static const QString maxFreq{"maxFreqMHz"}; /*!< Maximum frequency in MHz */
    static const QString lock{"lockExternal"}; /*!< Whether to lock to external reference */
    static const QString outputs{"outputs"}; /*!< Number of outputs */
    static const QString mf{"multFactor"}; /*!< Multiplication factor */
    static const QString role{"role"}; /*!< Clock role assignment */
    static const QString tunable{"tunable"}; /*!< Whether clock is tunable */
    static const QString manualTune{"manualTune"}; /*!< Manual tuning enabled */
}

// Clock manager keys (from ClockManager)
namespace ClockManager {
    static const QString clockManager{"ClockManager"}; /*!< Clock manager settings key */
    static const QString hwClocks{"hwClocks"}; /*!< Hardware clocks array */
    static const QString clockKey{"key"}; /*!< Clock key identifier */
    static const QString clockOutput{"output"}; /*!< Clock output number */
    static const QString clockName{"name"}; /*!< Clock display name */
}

// AWG/Chirp source keys (from AWG base class)
namespace AWG {
    static const QString key{"AWG"}; /*!< AWG hardware type key */
    static const QString prot{"hasProtectionPulse"}; /*!< Whether AWG has protection pulse digital output */
    static const QString amp{"hasAmpEnablePulse"}; /*!< Whether AWG has amplifier enable pulse digital output */
    static const QString min{"minFreqMHz"}; /*!< Minimum frequency in MHz */
    static const QString max{"maxFreqMHz"}; /*!< Maximum frequency in MHz */
    static const QString rampOnly{"rampOnly"}; /*!< Whether AWG can only generate frequency ramps */
    static const QString rate{"sampleRateHz"}; /*!< Sample rate in Hz */
    static const QString samples{"maxSamples"}; /*!< Maximum number of samples per waveform */
    static const QString triggered{"triggered"}; /*!< Whether AWG is externally triggered */
    static const QString hashes{"wfmHashes"}; /*!< Waveform hashes array */
    static const QString wfmName{"name"}; /*!< Waveform name */
    static const QString wfmHash{"hash"}; /*!< Waveform hash */
}

// Pulse generator keys (from PulseGenerator base class)
namespace PGen {
    static const QString numChannels{"numChannels"}; /*!< Number of pulse generator channels */
    static const QString minWidth{"minWidth"}; /*!< Minimum pulse width */
    static const QString maxWidth{"maxWidth"}; /*!< Maximum pulse width */
    static const QString minDelay{"minDelay"}; /*!< Minimum pulse delay */
    static const QString maxDelay{"maxDelay"}; /*!< Maximum pulse delay */
    static const QString minRepRate{"minRepRateHz"}; /*!< Minimum repetition rate in Hz */
    static const QString maxRepRate{"maxRepRateHz"}; /*!< Maximum repetition rate in Hz */
    static const QString lockExternal{"lockExternal"}; /*!< Whether to lock to external reference */
    static const QString canDutyCycle{"canDutyCycle"}; /*!< Whether duty cycle mode is supported */
    static const QString canTrigger{"canTrigger"}; /*!< Whether external triggering is supported */
    static const QString dutyMax{"dutyMaxPulses"}; /*!< Maximum duty cycle pulses */
    static const QString canSyncToChannel{"canSyncToChannel"}; /*!< Whether channels can be synchronized */
    static const QString canDisableChannels{"canDisableChannels"}; /*!< Whether individual channels can be disabled */
    static const QString channels{"channels"}; /*!< Channel configuration array */
    static const QString chName{"name"}; /*!< Channel name */
    static const QString chRole{"role"}; /*!< Channel role assignment */
}

// Flow controller keys (from FlowController base class)
namespace Flow {
    static const QString flowChannels{"numChannels"}; /*!< Number of flow channels */
    static const QString interval{"intervalMs"}; /*!< Polling interval in milliseconds */
    static const QString pUnits{"pressureUnits"}; /*!< Pressure units string */
    static const QString pDec{"pressureDecimals"}; /*!< Pressure decimal places */
    static const QString pMax{"pressureMax"}; /*!< Maximum pressure value */
    static const QString channels{"channels"}; /*!< Channel configuration array */
    static const QString chUnits{"units"}; /*!< Channel flow units */
    static const QString chDecimals{"decimals"}; /*!< Channel decimal places */
    static const QString chMax{"max"}; /*!< Channel maximum flow */
    static const QString chName{"name"}; /*!< Channel name */
}

// Pressure controller keys (from PressureController base class)
namespace PController {
    static const QString min{"min"}; /*!< Minimum pressure value */
    static const QString max{"max"}; /*!< Maximum pressure value */
    static const QString decimals{"decimal"}; /*!< Pressure decimal places */
    static const QString units{"units"}; /*!< Pressure units string */
    static const QString readOnly{"readOnly"}; /*!< Whether controller is read-only */
    static const QString readInterval{"intervalMs"}; /*!< Polling interval in milliseconds */
    static const QString hasValve{"hasValve"}; /*!< Whether controller has gate valve */
}

// Temperature controller keys (from TemperatureController base class)
namespace TC {
    static const QString interval{"pollIntervalMs"}; /*!< Polling interval in milliseconds */
    static const QString numChannels{"numChannels"}; /*!< Number of temperature channels */
    static const QString channels{"channels"}; /*!< Channel configuration array */
    static const QString units{"units"}; /*!< Temperature units string */
    static const QString chName{"name"}; /*!< Channel name */
    static const QString enabled{"enabled"}; /*!< Channel enabled state */
    static const QString decimals{"decimal"}; /*!< Temperature decimal places */
}

} // namespace BC::Key

#endif // HARDWAREKEYS_H