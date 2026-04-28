#include "virtualioboard.h"
#include <hardware/core/hardwareregistration.h>

#include <QRandomGenerator>

// Register hardware implementation
REGISTER_HARDWARE_META(VirtualIOBoard, "Virtual IOBoard for Testing")

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
