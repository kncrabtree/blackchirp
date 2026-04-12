#include "virtuallifscope.h"
#include <hardware/core/hardwareregistration.h>

#include <QTimer>
#include <math.h>
#include <QRandomGenerator>

// Register hardware implementation
REGISTER_HARDWARE_META(VirtualLifScope, "Virtual LIF Scope for Testing")
REGISTER_HARDWARE_SETTINGS(VirtualLifScope,
    {BC::Key::Digi::numAnalogChannels, "Analog Channels", "Number of analog input channels", 2, 1, 128, HwSettingPriority::Required},
    {BC::Key::Digi::numDigitalChannels, "Digital Channels", "Number of digital input channels", 0, 0, 128, HwSettingPriority::Required},
    {BC::Key::Digi::hasAuxTriggerChannel, "Aux Trigger Channel", "Has auxiliary trigger input", true, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::minFullScale, "Min Full Scale (V)", "Minimum full-scale voltage range", 0.05, QVariant{}, QVariant{}, HwSettingPriority::Important},
    {BC::Key::Digi::maxFullScale, "Max Full Scale (V)", "Maximum full-scale voltage range", 2.0, QVariant{}, QVariant{}, HwSettingPriority::Important},
    {BC::Key::Digi::minVOffset, "Min V Offset (V)", "Minimum vertical offset", -2.0, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::maxVOffset, "Max V Offset (V)", "Maximum vertical offset", 2.0, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::isTriggered, "Externally Triggered", "Digitizer uses external trigger signal", true, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::minTrigDelay, "Min Trig Delay (us)", "Minimum trigger delay in microseconds", -10.0, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::maxTrigDelay, "Max Trig Delay (us)", "Maximum trigger delay in microseconds", 10.0, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::minTrigLevel, "Min Trig Level (V)", "Minimum trigger threshold voltage", -5.0, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::maxTrigLevel, "Max Trig Level (V)", "Maximum trigger threshold voltage", 5.0, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::canBlockAverage, "Block Average", "Supports block averaging mode", false, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::maxBytes, "Max Bytes/Point", "Maximum bytes per sample", 2, 1, 4, HwSettingPriority::Optional},
    {BC::Key::Digi::maxRecordLength, "Max Record Length", "Maximum record length in samples", 100000000, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::maxAverages, "Max Averages", "Maximum number of block averages", 10000, QVariant{}, QVariant{}, HwSettingPriority::Optional}
)
REGISTER_HARDWARE_ARRAY(VirtualLifScope, BC::Key::Digi::sampleRates,
    "Sample Rates", "Available digitizer sample rates", HwSettingPriority::Important)
REGISTER_HARDWARE_ARRAY_ENTRY(VirtualLifScope, BC::Key::Digi::sampleRates,
    {{BC::Key::Digi::srText, "78.125 MSa/s"}, {BC::Key::Digi::srValue, 2.5e9/32}})
REGISTER_HARDWARE_ARRAY_ENTRY(VirtualLifScope, BC::Key::Digi::sampleRates,
    {{BC::Key::Digi::srText, "156.25 MSa/s"}, {BC::Key::Digi::srValue, 2.5e9/16}})
REGISTER_HARDWARE_ARRAY_ENTRY(VirtualLifScope, BC::Key::Digi::sampleRates,
    {{BC::Key::Digi::srText, "312.5 MSa/s"}, {BC::Key::Digi::srValue, 2.5e9/8}})
REGISTER_HARDWARE_ARRAY_ENTRY(VirtualLifScope, BC::Key::Digi::sampleRates,
    {{BC::Key::Digi::srText, "625 MSa/s"}, {BC::Key::Digi::srValue, 2.5e9/4}})
REGISTER_HARDWARE_ARRAY_ENTRY(VirtualLifScope, BC::Key::Digi::sampleRates,
    {{BC::Key::Digi::srText, "1250 MSa/s"}, {BC::Key::Digi::srValue, 2.5e9/2}})
REGISTER_HARDWARE_ARRAY_ENTRY(VirtualLifScope, BC::Key::Digi::sampleRates,
    {{BC::Key::Digi::srText, "2500 MSa/s"}, {BC::Key::Digi::srValue, 2.5e9}})

VirtualLifScope::VirtualLifScope(const QString& label, QObject *parent) :
    LifScope(QString(VirtualLifScope::staticMetaObject.className()), label, parent)
{
    using namespace BC::Key::Digi;

    setDefault(canMultiRecord,false);
    setDefault(multiBlock,false);

    if(!containsArray(sampleRates))
        setArray(sampleRates,{
                     {{srText,"78.125 MSa/s"},{srValue,2.5e9/32}},
                     {{srText,"156.25 MSa/s"},{srValue,2.5e9/16}},
                     {{srText,"312.5 MSa/s"},{srValue,2.5e9/8}},
                     {{srText,"625 MSa/s"},{srValue,2.5e9/4}},
                     {{srText,"1250 MSa/s"},{srValue,2.5e9/2}},
                     {{srText,"2500 MSa/s"},{srValue,2.5e9}}
                 });

    save();
}

VirtualLifScope::~VirtualLifScope()
{

}

bool VirtualLifScope::testConnection()
{
    return true;
}

void VirtualLifScope::initialize()
{

    p_timer = new QTimer(this);
    p_timer->setInterval(200);
}


void VirtualLifScope::readWaveform()
{
    QVector<qint8> out;
    auto qr = QRandomGenerator::global();

    if(d_refEnabled)
        out.resize(2*d_recordLength*d_bytesPerPoint);
    else
        out.resize(d_recordLength*d_bytesPerPoint);

    for(int i=0; i<d_recordLength; i++)
    {
        if(d_bytesPerPoint == 1)
        {
            qint8 dat = qr->bounded(-128,127);
            out[i] = dat;
        }
        else
        {
            qint16 dat = qr->bounded(-32768,32767);
            qint8 datmsb = dat / 256;
            qint8 datlsb = dat % 256;
            if(d_byteOrder == DigitizerConfig::LittleEndian)
            {
                out[2*i] = datlsb;
                out[2*i+1] = datmsb;
            }
            else
            {
                out[2*i] = datmsb;
                out[2*i+1] = datlsb;
            }
        }
    }

    if(d_refEnabled)
    {
        for(int i = d_recordLength; i<2*d_recordLength; i++)
        {
            if(d_bytesPerPoint == 1)
            {
                qint8 dat = qr->bounded(-128,127);
                out[i] = dat;
            }
            else
            {
                qint16 dat = qr->bounded(-32768,32767);
                qint8 datmsb = dat / 256;
                qint8 datlsb = dat % 256;
                if(d_byteOrder == DigitizerConfig::LittleEndian)
                {
                   out[2*i] = datlsb;
                   out[2*i+1] = datmsb;
                }
                else
                {
                    out[2*i] = datmsb;
                    out[2*i+1] = datlsb;
                }
            }
        }
    }

    emitWaveform(out);

}

bool VirtualLifScope::configure(const LifDigitizerConfig &c)
{
    static_cast<LifDigitizerConfig&>(*this) = c;
    d_channelOrder = Sequential;
    return true;
}

void VirtualLifScope::beginAcquisition()
{
    connect(p_timer,&QTimer::timeout,this,&VirtualLifScope::readWaveform);
    p_timer->start();
}

void VirtualLifScope::endAcquisition()
{
    p_timer->stop();
    disconnect(p_timer,&QTimer::timeout,this,&VirtualLifScope::readWaveform);
}
