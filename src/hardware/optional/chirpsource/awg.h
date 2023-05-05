#ifndef AWG_H
#define AWG_H

#include <hardware/core/hardwareobject.h>

namespace BC::Key::AWG {
static const QString key{"AWG"};
static const QString prot{"hasProtectionPulse"};
static const QString amp{"hasAmpEnablePulse"};
static const QString min{"minFreqMHz"};
static const QString max{"maxFreqMHz"};
static const QString rampOnly{"rampOnly"};
static const QString rate{"sampleRateHz"};
static const QString samples{"maxSamples"};
static const QString triggered{"triggered"};
static const QString hashes{"wfmHashes"};
static const QString wfmName{"name"};
static const QString wfmHash{"hash"};
}

/**
 * @brief The AWG class
 *
 * Interface class the CP-FTMW waveform generator.
 * This could be a true AWG or a ramp generator.
 *
 * Implementations of this class must define their hardware limits in QSettings
 * using the appropriate key/subKey labeling so that the UI can be configured
 * properly according to the device's capabilities.
 *
 * The following keys are available for use, and should be defined.
 * Implementations are free to choose whether these settings should be able to be modified by the user
 * or overwrittten on each program start.
 *
 * sampleRate - The sample clock frequency in Hz.
 * maxSamples - The maximum number of samples that can be stored in a single waveform. For an AWD, this is usually limited by the total installed memory. This setting is not used for ramp generators
 * minFreq - The minimum frequency waveform that can be generated, in MHz
 * maxFreq - The maximum frequency waveform that can be generated, in MHz
 * hasProtectionPulse - A boolean that indicates whether the AWG has a digital output to trigger a protection switch. If true, the program can guarantee that protection is always asserted when a chirp is active, and the user can configure its timing when generating the chirp
 * hasAmpEnablePulse - A boolean that indicates whether the AWG has a digital output to trigger an amplifier gate (e.g., the gate pulse on a TWT). If true, gate settings can be configured through the UI,
 * triggered - (optional) A boolean that indicates whether the AWG is externally triggered. Currently only used in AWG7122B.
 * rampOnly - A boolean that indicates if the AWG can ONLY generate a frequency ramp. If true, the RampConfig class will be used to configure the ramp rather than the ChirpConfig class, and a different UI will be employed for its configuration.
 */

class AWG : public HardwareObject
{
    Q_OBJECT
public:
    AWG(const QString subKey, const QString name, CommunicationProtocol::CommType commType,
        QObject *parent = nullptr, bool threaded = false, bool critical = true);
    virtual ~AWG();
};

#ifdef BC_AWG
#include BC_STR(BC_AWG_H)
#endif

#endif // AWG_H
