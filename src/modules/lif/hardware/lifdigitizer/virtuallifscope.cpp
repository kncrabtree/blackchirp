#include "virtuallifscope.h"

#include <QTimer>
#include <math.h>
#include <QRandomGenerator>


VirtualLifScope::VirtualLifScope(QObject *parent) :
    LifScope(BC::Key::Comm::hwVirtual,BC::Key::vLifScopeName,CommunicationProtocol::Virtual,parent)
{
    using namespace BC::Key::Digi;

    setDefault(numAnalogChannels,2);
    setDefault(numDigitalChannels,0);
    setDefault(hasAuxTriggerChannel,true);
    setDefault(minFullScale,5e-2);
    setDefault(maxFullScale,2.0);
    setDefault(minVOffset,-2.0);
    setDefault(maxVOffset,2.0);
    setDefault(isTriggered,true);
    setDefault(minTrigDelay,-10.0);
    setDefault(maxTrigDelay,10.0);
    setDefault(minTrigLevel,-5.0);
    setDefault(maxTrigLevel,5.0);
    setDefault(canBlockAverage,false);
    setDefault(canMultiRecord,false);
    setDefault(multiBlock,false);
    setDefault(maxBytes,2);

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

    emit waveformRead(out);

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
