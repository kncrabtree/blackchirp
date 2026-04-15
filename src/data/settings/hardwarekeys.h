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
    inline constexpr QLatin1StringView key{"key"}; /*!< Hardware type key */
    inline constexpr QLatin1StringView model{"model"}; /*!< Hardware implementation/model name */
    inline constexpr QLatin1StringView name{"prettyName"}; /*!< Display name */
    inline constexpr QLatin1StringView connected{"connected"}; /*!< Whether last communication was successful */
    inline constexpr QLatin1StringView critical{"critical"}; /*!< Whether communication failure should abort an experiment */
inline constexpr QLatin1StringView commType{"commType"}; /*!< CommunicationProtocol type */
    inline constexpr QLatin1StringView supportedProtocols{"supportedProtocols"}; /*!< List of supported communication protocols */
    inline constexpr QLatin1StringView rInterval{"rollingDataIntervalSec"}; /*!< Timer interval for rolling data (seconds) */
}

// Communication protocol keys (from CommunicationProtocol)
namespace Comm {
    inline constexpr QLatin1StringView rs232{"rs232"}; /*!< Rs232Instrument */
    inline constexpr QLatin1StringView tcp{"tcp"}; /*!< TcpInstrument */
    inline constexpr QLatin1StringView gpib{"gpib"}; /*!< GpibInstrument */
    inline constexpr QLatin1StringView custom{"custom"}; /*!< CustomInstrument */
    inline constexpr QLatin1StringView hwVirtual{"virtual"}; /*!< VirtualInstrument */
    inline constexpr QLatin1StringView timeout{"timeout"}; /*!< Read timeout in ms (<=0 disables timeout) */
    inline constexpr QLatin1StringView termChar{"termChar"}; /*!< Termination character(s) (empty disables termChar) */
}

// RS232 protocol specific keys (from Rs232Instrument)
namespace RS232 {
    inline constexpr QLatin1StringView baud{"baudrate"};
    inline constexpr QLatin1StringView id{"id"};
    inline constexpr QLatin1StringView dataBits{"databits"};
    inline constexpr QLatin1StringView parity{"parity"};
    inline constexpr QLatin1StringView stopBits{"stopbits"};
    inline constexpr QLatin1StringView flowControl{"flowControl"};
}

// TCP protocol specific keys (from TcpInstrument)
namespace TCP {
    inline constexpr QLatin1StringView ip{"ip"};
    inline constexpr QLatin1StringView port{"port"};
}

// GPIB protocol specific keys (from GpibInstrument)
namespace GPIB {
    inline constexpr QLatin1StringView gpibAddress{"address"};
    inline constexpr QLatin1StringView gpibController{"controller"}; // Future support for multiple controllers
}

// Custom protocol specific keys (from CustomInstrument)
namespace Custom {
    inline constexpr QLatin1StringView comm{"comm"}; /*!< Key for communication array. */
    inline constexpr QLatin1StringView type{"type"}; /*!< Type of data entry field (stringKey or intKey). */
    inline constexpr QLatin1StringView key{"key"}; /*!< Identifier key for this data field. */
    inline constexpr QLatin1StringView intKey{"int"}; /*!< Key for integer data entry type. */
    inline constexpr QLatin1StringView intMin{"min"}; /*!< Minimum allowed integer value. */
    inline constexpr QLatin1StringView intMax{"max"}; /*!< Maximum allowed integer value. */
    inline constexpr QLatin1StringView stringKey{"string"}; /*!< Key for string data entry type. */
    inline constexpr QLatin1StringView maxLen{"length"}; /*!< Maximum allowed length of string. */
    inline constexpr QLatin1StringView label{"name"}; /*!< Label displayed to user for this data field. */
}

// FTMW digitizer keys (from FtmwScope)
namespace FtmwScope {
    inline constexpr QLatin1StringView ftmwScope{"FtmwDigitizer"}; /*!< FTMW digitizer hardware type key */
    inline constexpr QLatin1StringView bandwidth{"bandwidthMHz"}; /*!< Analog bandwidth in MHz */
    inline constexpr QLatin1StringView fidCh{"fidChannel"}; /*!< FID recording channel */
}

// Digitizer configuration keys (from DigitizerConfig)
namespace Digi {
    inline constexpr QLatin1StringView dwAnChannels{"channels"}; /*!< Analog channels configuration array */
    inline constexpr QLatin1StringView dwDigChannels{"digitalChannels"}; /*!< Digital channels configuration array */
    inline constexpr QLatin1StringView numAnalogChannels{"numAnalogChannels"}; /*!< Number of analog channels */
    inline constexpr QLatin1StringView hasAuxTriggerChannel{"hasAuxTriggerChannel"}; /*!< Whether auxiliary trigger channel exists */
    inline constexpr QLatin1StringView numDigitalChannels{"numDigitalChannels"}; /*!< Number of digital channels */
    inline constexpr QLatin1StringView minFullScale{"minFullScale"}; /*!< Minimum full scale voltage */
    inline constexpr QLatin1StringView maxFullScale{"maxFullScale"}; /*!< Maximum full scale voltage */
    inline constexpr QLatin1StringView minVOffset{"minVOffset"}; /*!< Minimum vertical offset */
    inline constexpr QLatin1StringView maxVOffset{"maxVOffset"}; /*!< Maximum vertical offset */
    inline constexpr QLatin1StringView isTriggered{"isTriggered"}; /*!< Whether digitizer supports triggering */
    inline constexpr QLatin1StringView minTrigDelay{"minTrigDelayUs"}; /*!< Minimum trigger delay in microseconds */
    inline constexpr QLatin1StringView maxTrigDelay{"maxTrigDelayUs"}; /*!< Maximum trigger delay in microseconds */
    inline constexpr QLatin1StringView minTrigLevel{"minTrigLevel"}; /*!< Minimum trigger level */
    inline constexpr QLatin1StringView maxTrigLevel{"maxTrigLevel"}; /*!< Maximum trigger level */
    inline constexpr QLatin1StringView maxRecordLength{"maxRecordLength"}; /*!< Maximum record length */
    inline constexpr QLatin1StringView canMultiRecord{"canMultiRecord"}; /*!< Whether multi-record mode is supported */
    inline constexpr QLatin1StringView maxRecords{"maxRecords"}; /*!< Maximum number of records */
    inline constexpr QLatin1StringView canBlockAverage{"canBlockAverage"}; /*!< Whether block averaging is supported */
    inline constexpr QLatin1StringView maxAverages{"maxAverages"}; /*!< Maximum number of averages */
    inline constexpr QLatin1StringView multiBlock{"canBlockAndMultiRecord"}; /*!< Whether block and multi-record modes can be combined */
    inline constexpr QLatin1StringView maxBytes{"maxBytesPerPoint"}; /*!< Maximum bytes per data point */
    inline constexpr QLatin1StringView sampleRates{"sampleRates"}; /*!< Supported sample rates array */
    inline constexpr QLatin1StringView srText{"text"}; /*!< Sample rate display text */
    inline constexpr QLatin1StringView srValue{"val"}; /*!< Sample rate numeric value */
}

// Clock configuration keys (from Clock base class)
namespace Clock {
    inline constexpr QLatin1StringView clock{"Clock"}; /*!< Clock hardware type key */
    inline constexpr QLatin1StringView minFreq{"minFreqMHz"}; /*!< Minimum frequency in MHz */
    inline constexpr QLatin1StringView maxFreq{"maxFreqMHz"}; /*!< Maximum frequency in MHz */
    inline constexpr QLatin1StringView lock{"lockExternal"}; /*!< Whether to lock to external reference */
    inline constexpr QLatin1StringView outputs{"outputs"}; /*!< Number of outputs */
    inline constexpr QLatin1StringView mf{"multFactor"}; /*!< Multiplication factor */
    inline constexpr QLatin1StringView role{"role"}; /*!< Clock role assignment */
    inline constexpr QLatin1StringView tunable{"tunable"}; /*!< Whether clock is tunable */
    inline constexpr QLatin1StringView manualTune{"manualTune"}; /*!< Manual tuning enabled */
}

// Clock manager keys (from ClockManager)
namespace ClockManager {
    inline constexpr QLatin1StringView clockManager{"ClockManager"}; /*!< Clock manager settings key */
    inline constexpr QLatin1StringView hwClocks{"hwClocks"}; /*!< Hardware clocks array */
    inline constexpr QLatin1StringView clockKey{"key"}; /*!< Clock key identifier */
    inline constexpr QLatin1StringView clockOutput{"output"}; /*!< Clock output number */
    inline constexpr QLatin1StringView clockName{"name"}; /*!< Clock display name */
}

// AWG/Chirp source keys (from AWG base class)
namespace AWG {
    inline constexpr QLatin1StringView key{"AWG"}; /*!< AWG hardware type key */
    inline constexpr QLatin1StringView markerCount{"markerCount"}; /*!< Number of physical marker output channels */
    inline constexpr QLatin1StringView min{"minFreqMHz"}; /*!< Minimum frequency in MHz */
    inline constexpr QLatin1StringView max{"maxFreqMHz"}; /*!< Maximum frequency in MHz */
    inline constexpr QLatin1StringView rampOnly{"rampOnly"}; /*!< Whether AWG can only generate frequency ramps */
    inline constexpr QLatin1StringView rate{"sampleRateHz"}; /*!< Sample rate in Hz */
    inline constexpr QLatin1StringView samples{"maxSamples"}; /*!< Maximum number of samples per waveform */
    inline constexpr QLatin1StringView triggered{"triggered"}; /*!< Whether AWG is externally triggered */
    inline constexpr QLatin1StringView hashes{"wfmHashes"}; /*!< Waveform hashes array */
    inline constexpr QLatin1StringView wfmName{"name"}; /*!< Waveform name */
    inline constexpr QLatin1StringView wfmHash{"hash"}; /*!< Waveform hash */
}

// Pulse generator keys (from PulseGenerator base class)
namespace PGen {
    inline constexpr QLatin1StringView numChannels{"numChannels"}; /*!< Number of pulse generator channels */
    inline constexpr QLatin1StringView minWidth{"minWidth"}; /*!< Minimum pulse width */
    inline constexpr QLatin1StringView maxWidth{"maxWidth"}; /*!< Maximum pulse width */
    inline constexpr QLatin1StringView minDelay{"minDelay"}; /*!< Minimum pulse delay */
    inline constexpr QLatin1StringView maxDelay{"maxDelay"}; /*!< Maximum pulse delay */
    inline constexpr QLatin1StringView minRepRate{"minRepRateHz"}; /*!< Minimum repetition rate in Hz */
    inline constexpr QLatin1StringView maxRepRate{"maxRepRateHz"}; /*!< Maximum repetition rate in Hz */
    inline constexpr QLatin1StringView lockExternal{"lockExternal"}; /*!< Whether to lock to external reference */
    inline constexpr QLatin1StringView canDutyCycle{"canDutyCycle"}; /*!< Whether duty cycle mode is supported */
    inline constexpr QLatin1StringView canTrigger{"canTrigger"}; /*!< Whether external triggering is supported */
    inline constexpr QLatin1StringView dutyMax{"dutyMaxPulses"}; /*!< Maximum duty cycle pulses */
    inline constexpr QLatin1StringView canSyncToChannel{"canSyncToChannel"}; /*!< Whether channels can be synchronized */
    inline constexpr QLatin1StringView canDisableChannels{"canDisableChannels"}; /*!< Whether individual channels can be disabled */
    inline constexpr QLatin1StringView channels{"channels"}; /*!< Channel configuration array */
    inline constexpr QLatin1StringView chName{"name"}; /*!< Channel name */
    inline constexpr QLatin1StringView chRole{"role"}; /*!< Channel role assignment */
}

// Flow controller keys (from FlowController base class)
namespace Flow {
    inline constexpr QLatin1StringView flowChannels{"numChannels"}; /*!< Number of flow channels */
    inline constexpr QLatin1StringView interval{"intervalMs"}; /*!< Polling interval in milliseconds */
    inline constexpr QLatin1StringView pUnits{"pressureUnits"}; /*!< Pressure units string */
    inline constexpr QLatin1StringView pDec{"pressureDecimals"}; /*!< Pressure decimal places */
    inline constexpr QLatin1StringView pMax{"pressureMax"}; /*!< Maximum pressure value */
    inline constexpr QLatin1StringView channels{"channels"}; /*!< Channel configuration array */
    inline constexpr QLatin1StringView chUnits{"units"}; /*!< Channel flow units */
    inline constexpr QLatin1StringView chDecimals{"decimals"}; /*!< Channel decimal places */
    inline constexpr QLatin1StringView chMax{"max"}; /*!< Channel maximum flow */
    inline constexpr QLatin1StringView chName{"name"}; /*!< Channel name */
}

// Pressure controller keys (from PressureController base class)
namespace PController {
    inline constexpr QLatin1StringView min{"min"}; /*!< Minimum pressure value */
    inline constexpr QLatin1StringView max{"max"}; /*!< Maximum pressure value */
    inline constexpr QLatin1StringView decimals{"decimal"}; /*!< Pressure decimal places */
    inline constexpr QLatin1StringView units{"units"}; /*!< Pressure units string */
    inline constexpr QLatin1StringView readOnly{"readOnly"}; /*!< Whether controller is read-only */
    inline constexpr QLatin1StringView readInterval{"intervalMs"}; /*!< Polling interval in milliseconds */
    inline constexpr QLatin1StringView hasValve{"hasValve"}; /*!< Whether controller has gate valve */
}

// Temperature controller keys (from TemperatureController base class)
namespace TC {
    inline constexpr QLatin1StringView interval{"pollIntervalMs"}; /*!< Polling interval in milliseconds */
    inline constexpr QLatin1StringView numChannels{"numChannels"}; /*!< Number of temperature channels */
    inline constexpr QLatin1StringView channels{"channels"}; /*!< Channel configuration array */
    inline constexpr QLatin1StringView units{"units"}; /*!< Temperature units string */
    inline constexpr QLatin1StringView chName{"name"}; /*!< Channel name */
    inline constexpr QLatin1StringView enabled{"enabled"}; /*!< Channel enabled state */
    inline constexpr QLatin1StringView decimals{"decimal"}; /*!< Temperature decimal places */
}

} // namespace BC::Key

#endif // HARDWAREKEYS_H