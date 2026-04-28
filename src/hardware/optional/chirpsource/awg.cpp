#include <hardware/optional/chirpsource/awg.h>
#include <hardware/core/hardwareregistration.h>

REGISTER_HARDWARE_BASE(AWG,
    {BC::Key::AWG::rate, "Sample Rate (Hz)", "DAC output sample rate",
     16e9, 1e6, 1000e9, HwSettingPriority::Important},
    {BC::Key::AWG::samples, "Max Samples", "Maximum waveform sample count",
     2e9, 0, QVariant{}, HwSettingPriority::Important},
    {BC::Key::AWG::min, "Min Freq (MHz)", "Minimum chirp frequency in MHz",
     100.0, 0.0, QVariant{}, HwSettingPriority::Important},
    {BC::Key::AWG::max, "Max Freq (MHz)", "Maximum chirp frequency in MHz",
     6250.0, 0.0, QVariant{}, HwSettingPriority::Important},
    {BC::Key::AWG::markerCount, "Marker Count", "Number of physical marker output channels",
     4, 0, QVariant{}, HwSettingPriority::Required},
    {BC::Key::AWG::rampOnly, "Ramp Only", "Restrict to linear frequency ramp chirps (no arbitrary waveforms)",
     false, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::AWG::triggered, "Triggered", "AWG waits for an external trigger before outputting",
     true, QVariant{}, QVariant{}, HwSettingPriority::Optional}
)

AWG::AWG(const QString& impl, const QString& label, QObject *parent) :
    HardwareObject(QString(AWG::staticMetaObject.className()), impl, label, parent)
{
    d_threaded = true;
}

AWG::~AWG()
{

}
