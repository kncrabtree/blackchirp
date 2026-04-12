#include "virtualioboard.h"
#include <hardware/core/hardwareregistration.h>

#include <QRandomGenerator>

// Register hardware implementation
REGISTER_HARDWARE_META(VirtualIOBoard, "Virtual IOBoard for Testing")
REGISTER_HARDWARE_SETTINGS(VirtualIOBoard,
    {BC::Key::Digi::numAnalogChannels, "Analog Channels", "Number of analog input channels",
     8, 0, 128, HwSettingPriority::Required},
    {BC::Key::Digi::numDigitalChannels, "Digital Channels", "Number of digital input channels",
     8, 0, 128, HwSettingPriority::Required},
    {BC::Key::Digi::hasAuxTriggerChannel, "Aux Trigger Channel", "Has auxiliary trigger input",
     false, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::minFullScale, "Min Full Scale (V)", "Minimum full-scale voltage range",
     2.44, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::maxFullScale, "Max Full Scale (V)", "Maximum full-scale voltage range",
     2.44, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::minVOffset, "Min V Offset (V)", "Minimum vertical offset",
     0.0, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::maxVOffset, "Max V Offset (V)", "Maximum vertical offset",
     0.0, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::minTrigDelay, "Min Trig Delay (us)", "Minimum trigger delay in microseconds",
     0.0, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::maxTrigDelay, "Max Trig Delay (us)", "Maximum trigger delay in microseconds",
     0.0, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::minTrigLevel, "Min Trig Level (V)", "Minimum trigger threshold voltage",
     0.0, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::maxTrigLevel, "Max Trig Level (V)", "Maximum trigger threshold voltage",
     0.0, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::maxRecordLength, "Max Record Length", "Maximum record length in samples",
     1, 1, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::canBlockAverage, "Block Average", "Supports block averaging",
     false, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::canMultiRecord, "Multi Record", "Supports multi-record acquisition",
     false, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::multiBlock, "Multi Block", "Can simultaneously block average and multi-record",
     false, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::maxBytes, "Max Bytes/Point", "Maximum bytes per sample",
     2, 1, 8, HwSettingPriority::Optional}
)
REGISTER_HARDWARE_ARRAY(VirtualIOBoard, BC::Key::Digi::sampleRates,
    "Sample Rates", "Available sample rates (for IO boards typically 'N/A')",
    HwSettingPriority::Optional)
REGISTER_HARDWARE_ARRAY_ENTRY(VirtualIOBoard, BC::Key::Digi::sampleRates,
    {{BC::Key::Digi::srText, "N/A"}, {BC::Key::Digi::srValue, 0.0}})

VirtualIOBoard::VirtualIOBoard(const QString& label, QObject *parent) :
    IOBoard(QString(VirtualIOBoard::staticMetaObject.className()), label, parent)
{
    using namespace BC::Key::Digi;

    if(!containsArray(sampleRates))
        setArray(sampleRates,{
                     {{srText,"N/A"},{srValue,0.0}},
                 });

    save();
}




bool VirtualIOBoard::testConnection()
{
    return true;
}

void VirtualIOBoard::initialize()
{
}


bool VirtualIOBoard::configure(IOBoardConfig &config)
{
    Q_UNUSED(config)
    return true;
}

std::map<int, double> VirtualIOBoard::readAnalogChannels()
{
    auto qr = QRandomGenerator::global();
    std::map<int,double> out;
    for(auto const &[k,ch] : d_analogChannels)
    {
        if(ch.enabled)
        {
            out.insert({k,static_cast<double>(qr->bounded(0,2 << d_bytesPerPoint*8))/
                        (2 << d_bytesPerPoint*8)*ch.fullScale});
        }
    }

    return out;
}

std::map<int, bool> VirtualIOBoard::readDigitalChannels()
{
    auto qr = QRandomGenerator::global();
    std::map<int,bool> out;
    for(auto const &[k,ch] : d_digitalChannels)
    {
        if(ch.enabled)
            out.insert({k,static_cast<bool>(qr->bounded(1))});
    }

    return out;
}
