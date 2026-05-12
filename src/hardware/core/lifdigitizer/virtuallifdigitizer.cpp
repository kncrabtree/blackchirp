#include "virtuallifdigitizer.h"
#include <hardware/core/hardwareregistration.h>

#include <QTimer>
#include <math.h>
#include <QRandomGenerator>

// Register hardware implementation
REGISTER_HARDWARE_META(VirtualLifDigitizer, "Virtual LIF Scope for Testing")
REGISTER_HARDWARE_ARRAY(VirtualLifDigitizer, BC::Key::Digi::sampleRates,
    "Sample Rates", "Available digitizer sample rates", HwSettingPriority::Important)
REGISTER_HARDWARE_ARRAY_ENTRY(VirtualLifDigitizer, BC::Key::Digi::sampleRates,
    {{BC::Key::Digi::srText, "78.125 MSa/s"}, {BC::Key::Digi::srValue, 2.5e9/32}})
REGISTER_HARDWARE_ARRAY_ENTRY(VirtualLifDigitizer, BC::Key::Digi::sampleRates,
    {{BC::Key::Digi::srText, "156.25 MSa/s"}, {BC::Key::Digi::srValue, 2.5e9/16}})
REGISTER_HARDWARE_ARRAY_ENTRY(VirtualLifDigitizer, BC::Key::Digi::sampleRates,
    {{BC::Key::Digi::srText, "312.5 MSa/s"}, {BC::Key::Digi::srValue, 2.5e9/8}})
REGISTER_HARDWARE_ARRAY_ENTRY(VirtualLifDigitizer, BC::Key::Digi::sampleRates,
    {{BC::Key::Digi::srText, "625 MSa/s"}, {BC::Key::Digi::srValue, 2.5e9/4}})
REGISTER_HARDWARE_ARRAY_ENTRY(VirtualLifDigitizer, BC::Key::Digi::sampleRates,
    {{BC::Key::Digi::srText, "1250 MSa/s"}, {BC::Key::Digi::srValue, 2.5e9/2}})
REGISTER_HARDWARE_ARRAY_ENTRY(VirtualLifDigitizer, BC::Key::Digi::sampleRates,
    {{BC::Key::Digi::srText, "2500 MSa/s"}, {BC::Key::Digi::srValue, 2.5e9}})

VirtualLifDigitizer::VirtualLifDigitizer(const QString& label, QObject *parent) :
    LifDigitizer(QString(VirtualLifDigitizer::staticMetaObject.className()), label, parent)
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

VirtualLifDigitizer::~VirtualLifDigitizer()
{

}

bool VirtualLifDigitizer::testConnection()
{
    return true;
}

void VirtualLifDigitizer::initialize()
{

    p_timer = new QTimer(this);
    p_timer->setInterval(200);
}


void VirtualLifDigitizer::readWaveform()
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

bool VirtualLifDigitizer::configure(const LifDigitizerConfig &c)
{
    static_cast<LifDigitizerConfig&>(*this) = c;
    d_channelOrder = Sequential;
    return true;
}

void VirtualLifDigitizer::beginAcquisition()
{
    connect(p_timer,&QTimer::timeout,this,&VirtualLifDigitizer::readWaveform);
    p_timer->start();
}

void VirtualLifDigitizer::endAcquisition()
{
    p_timer->stop();
    disconnect(p_timer,&QTimer::timeout,this,&VirtualLifDigitizer::readWaveform);
}
