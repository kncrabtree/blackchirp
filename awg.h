#ifndef AWG_H
#define AWG_H

#include "hardwareobject.h"

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
    AWG(QObject *parent);
    virtual ~AWG();
};


#ifdef BC_AWG
#if BC_AWG==1
#include "awg70002a.h"
class AWG70002a;
typedef AWG70002a AwgHardware;
#elif BC_AWG==2
#include "awg7122b.h"
class AWG7122B;
typedef AWG7122B AwgHardware;
#elif BC_AWG==3
#include "ad9914.h"
class AD9914;
typedef AD9914 AwgHardware;
#elif BC_AWG==4
#include "m8195a.h"
class M8195A;
typedef M8195A AwgHardware;
#else
#include "virtualawg.h"
class VirtualAwg;
typedef VirtualAwg AwgHardware;
#endif
#endif

#endif // AWG_H
