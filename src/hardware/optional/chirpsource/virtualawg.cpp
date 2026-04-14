#include "virtualawg.h"
#include <hardware/core/hardwareregistration.h>

// Register hardware implementation
REGISTER_HARDWARE_META(VirtualAwg, "Virtual AWG for testing and simulation")
REGISTER_HARDWARE_SETTINGS(VirtualAwg,
    {BC::Key::AWG::rate,      "Sample Rate (Hz)", "DAC output sample rate",
     16e9, 1e6, 100e9, HwSettingPriority::Important},
    {BC::Key::AWG::samples,   "Max Samples",      "Maximum waveform sample count",
     2e9, 0.0, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::AWG::min,       "Min Freq (MHz)",    "Minimum chirp frequency",
     100.0, 0.0, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::AWG::max,       "Max Freq (MHz)",    "Maximum chirp frequency",
     6250.0, 0.0, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::AWG::markerCount, "Marker Count", "Number of physical marker output channels",
     4, 0, QVariant{}, HwSettingPriority::Required},
    {BC::Key::AWG::rampOnly,  "Ramp Only",         "Restrict to linear ramp chirps",
     false, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::AWG::triggered, "Triggered",         "AWG waits for external trigger",
     true, QVariant{}, QVariant{}, HwSettingPriority::Optional}
)

VirtualAwg::VirtualAwg(const QString& label, QObject *parent) :
    AWG(QString(VirtualAwg::staticMetaObject.className()), label, parent)
{
}

VirtualAwg::~VirtualAwg()
{

}

bool VirtualAwg::testConnection()
{
    return true;
}

void VirtualAwg::initialize()
{
}
