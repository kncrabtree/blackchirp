#include "virtualftmwscope.h"

#include <QRandomGenerator>
#include <math.h>
#include <hardware/core/hardwareregistration.h>

using namespace BC::Key::FtmwScope;
using namespace BC::Key::Digi;

// Register this hardware implementation
REGISTER_HARDWARE_META(VirtualFtmwScope, "Virtual FTMW digitizer for testing and development")
REGISTER_HARDWARE_PROTOCOLS(VirtualFtmwScope, CommunicationProtocol::Virtual,
    CommunicationProtocol::Rs232, CommunicationProtocol::Tcp,
    CommunicationProtocol::Gpib, CommunicationProtocol::Custom)

VirtualFtmwScope::VirtualFtmwScope(const QString& label, QObject *parent) :
    FtmwScope(QString(VirtualFtmwScope::staticMetaObject.className()), label, parent)
{
    setDefault(numAnalogChannels,4);
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
    setDefault(maxRecordLength,100000000);
    setDefault(canBlockAverage,true);
    setDefault(maxAverages,100);
    setDefault(canMultiRecord,true);
    setDefault(maxRecords,100);
    setDefault(multiBlock,false);
    setDefault(maxBytes,2);
    setDefault(bandwidth,16000.0);

    if(!containsArray(sampleRates))
        setArray(sampleRates,{
                     {{srText,"2 GSa/s"},{srValue,2e9}},
                       {{srText,"5 GSa/s"},{srValue,5e9}},
                       {{srText,"10 GSa/s"},{srValue,10e9}},
                       {{srText,"20 GSa/s"},{srValue,20e9}},
                       {{srText,"50 GSa/s"},{srValue,50e9}},
                       {{srText,"100 GSa/s"},{srValue,100e9}}
                     });
    save();
}

VirtualFtmwScope::~VirtualFtmwScope()
{

}

bool VirtualFtmwScope::testConnection()
{
    d_simulatedTimer->stop();
    d_simulatedTimer->setInterval(get(interval,200));

    return true;
}

void VirtualFtmwScope::initialize()
{
    // Initialize simulated data as empty - will be populated in prepareForExperiment
    d_simulatedData.clear();

    d_simulatedTimer = new QTimer(this);
    connect(d_simulatedTimer,&QTimer::timeout,this,&FtmwScope::readWaveform, Qt::UniqueConnection);
}

bool VirtualFtmwScope::prepareForExperiment(Experiment &exp)
{
    //make a copy of the configuration in which to store settings
    if(!exp.ftmwEnabled())
        return true;

    static_cast<FtmwDigitizerConfig&>(*this) = exp.ftmwConfig()->scopeConfig();
    
    // Generate simulated FID data based on experiment configuration
    generateSimulatedFid();
    
    return true;

}

void VirtualFtmwScope::beginAcquisition()
{
    d_simulatedTimer->start();
}

void VirtualFtmwScope::endAcquisition()
{
    d_simulatedTimer->stop();
}

void VirtualFtmwScope::readWaveform()
{
    //    d_testTime.restart();
        QByteArray out;

        int frames = 1;
        if(d_multiRecord)
        {
            frames = d_numRecords;
            out.resize(d_recordLength*d_bytesPerPoint*frames);
        }
        else
            out.resize(d_recordLength*d_bytesPerPoint);


        double ym = yMult(d_fidChannel);
        for(int i=0; i<frames; i++)
        {
            for(int j=0; j<d_recordLength; j++)
            {
                //using the value function here because j could exceed simulated data size
                double dat = d_simulatedData.value(j);


                if(d_bytesPerPoint == 1)
                {
                    int noise = (rand()%32)-16;
                    qint8 n = qBound(-128,((int)(dat/ym)+noise),127);
                    out[d_recordLength*i + j] = n;
                }
                else
                {
                    int noise = (rand()%4096)-2048;
                    qint16 n = qBound(-32768,((int)(dat/ym)+noise),32767);
                    quint8 byte1;
                    quint8 byte2;
                    if(d_byteOrder == DigitizerConfig::LittleEndian)
                    {
                        byte1 = (n & 0x00ff);
                        byte2 = (n & 0xff00) >> 8;
                    }
                    else
                    {
                        byte1 = (n & 0xff00) >> 8;
                        byte2 = (n & 0x00ff);
                    }
                    out[d_recordLength*2*i + 2*j] = byte1;
                    out[d_recordLength*2*i + 2*j + 1] = byte2;
                }
            }
        }
    //    emit logMessage(QString("Simulate: %1 ms").arg(d_testTime.elapsed()));
        emitShot(out);
}


void VirtualFtmwScope::generateSimulatedFid()
{
    // Clear any existing data
    d_simulatedData.clear();
    d_simulatedData.resize(d_recordLength);
    d_simulatedData.fill(0.0);
    
    // Get full scale for amplitude scaling
    double fullScale = d_analogChannels.at(d_fidChannel).fullScale;
    
    // Generate random number of frequency components (10-100)
    QRandomGenerator *rng = QRandomGenerator::global();
    int numComponents = rng->bounded(10, 101);
    
    // Calculate time step and Nyquist frequency
    double dt = 1.0 / d_sampleRate;
    double nyquistFreq = d_sampleRate / 2.0;
    
    // Damping time constant (about 1/2 record length)
    double dampingTime = (d_recordLength * dt) / 2.0;
    
    for(int comp = 0; comp < numComponents; comp++)
    {
        // Random frequency: 10% to 90% of Nyquist frequency
        double minFreq = nyquistFreq * 0.1;
        double maxFreq = nyquistFreq * 0.9;
        double frequency = rng->generateDouble() * (maxFreq - minFreq) + minFreq;
        
        // Random amplitude
        double minAmp = fullScale * 0.000001;
        double maxAmp = fullScale * 0.005;
        double amplitude = rng->generateDouble() * (maxAmp - minAmp) + minAmp;
        
        // Random phase
        double phase = rng->generateDouble() * 2.0 * M_PI;
        
        // Add this component to the FID
        for(int i = 0; i < d_recordLength; i++)
        {
            double t = i * dt;
            double decay = exp(-t / dampingTime);
            double signal = amplitude * decay * cos(2.0 * M_PI * frequency * t + phase);
            d_simulatedData[i] += signal;
        }
    }
}
