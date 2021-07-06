#include "virtualftmwscope.h"

#include <QFile>
#include <math.h>

using namespace BC::Key::FtmwScope;
using namespace BC::Key::Digi;

VirtualFtmwScope::VirtualFtmwScope(QObject *parent) :
    FtmwScope(BC::Key::hwVirtual,vftmwName,CommunicationProtocol::Virtual,parent)
{
    setDefault(numChannels,4);
    setDefault(hasAuxChannel,true);
    setDefault(minFullScale,5e-2);
    setDefault(maxFullScale,2.0);
    setDefault(minVOffset,-2.0);
    setDefault(maxVOffset,2.0);
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
    QFile f(QString(":/data/virtualdata.txt"));
    d_simulatedData.reserve(750000);
    if(f.open(QIODevice::ReadOnly))
    {
       while(!f.atEnd())
       {
          QByteArray l = f.readLine().trimmed();
          if(l.isEmpty())
             break;

          bool ok = false;
          double d = l.toDouble(&ok);
          if(ok)
             d_simulatedData.append(d);
          else
             d_simulatedData.append(0.0);
       }
       f.close();
    }
    else
    {
       for(int i=0;i<750000;i++)
          d_simulatedData.append(0.0);
    }

    d_simulatedTimer = new QTimer(this);
    connect(d_simulatedTimer,&QTimer::timeout,this,&FtmwScope::readWaveform, Qt::UniqueConnection);
}

bool VirtualFtmwScope::prepareForExperiment(Experiment &exp)
{
    //make a copy of the configuration in which to store settings
    if(!exp.ftmwEnabled())
        return true;

    static_cast<FtmwDigitizerConfig>(*this) = exp.ftmwConfig()->d_scopeConfig;
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


        for(int i=0; i<frames; i++)
        {
            for(int j=0; j<d_recordLength; j++)
            {
                //using the value function here because j could exceed simulated data size
                double dat = d_simulatedData.value(j);


                if(d_bytesPerPoint == 1)
                {
                    int noise = (rand()%32)-16;
                    qint8 n = qBound(-128,((int)(dat/yMult(d_fidChannel))+noise),127);
                    out[d_recordLength*i + j] = n;
                }
                else
                {
                    int noise = (rand()%4096)-2048;
                    qint16 n = qBound(-32768,((int)(dat/yMult(d_fidChannel))+noise),32767);
                    quint8 byte1;
                    quint8 byte2;
                    if(d_byteOrder == QDataStream::LittleEndian)
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
        emit shotAcquired(out);
}
