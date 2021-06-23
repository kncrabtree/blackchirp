#include "virtualftmwscope.h"

#include <QFile>
#include <math.h>

VirtualFtmwScope::VirtualFtmwScope(QObject *parent) :
    FtmwScope(BC::Key::hwVirtual,BC::Key::FtmwScope::vftmwName,CommunicationProtocol::Virtual,parent)
{
    setDefault(BC::Key::FtmwScope::blockAverage,false);
    setDefault(BC::Key::FtmwScope::multiRecord,true);
    setDefault(BC::Key::FtmwScope::summaryRecord,true);
    setDefault(BC::Key::FtmwScope::multiBlock,false);
    setDefault(BC::Key::FtmwScope::bandwidth,16000.0);

    if(!containsArray(BC::Key::FtmwScope::sampleRates))
        setArray(BC::Key::FtmwScope::sampleRates,{
                     {{BC::Key::FtmwScope::srText,"2 GSa/s"},{BC::Key::FtmwScope::srValue,2e9}},
                       {{BC::Key::FtmwScope::srText,"5 GSa/s"},{BC::Key::FtmwScope::srValue,5e9}},
                       {{BC::Key::FtmwScope::srText,"10 GSa/s"},{BC::Key::FtmwScope::srValue,10e9}},
                       {{BC::Key::FtmwScope::srText,"20 GSa/s"},{BC::Key::FtmwScope::srValue,20e9}},
                       {{BC::Key::FtmwScope::srText,"50 GSa/s"},{BC::Key::FtmwScope::srValue,50e9}},
                       {{BC::Key::FtmwScope::srText,"100 GSa/s"},{BC::Key::FtmwScope::srValue,100e9}}
                     });
}

VirtualFtmwScope::~VirtualFtmwScope()
{

}

bool VirtualFtmwScope::testConnection()
{
    d_simulatedTimer->stop();
    d_simulatedTimer->setInterval(get(BC::Key::FtmwScope::interval,200));

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
    if(!exp.ftmwConfig().isEnabled())
        return true;

    BlackChirp::FtmwScopeConfig config(exp.ftmwConfig().scopeConfig());

    config.yOff = 0;
    config.yMult = config.vScale*5.0/pow(2.0,8.0*config.bytesPerPoint-1.0);
    config.xIncr = 1.0/config.sampleRate;
    d_configuration = config;
    exp.setScopeConfig(config);
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
        if(d_configuration.fastFrameEnabled && !d_configuration.summaryFrame)
        {
            frames = d_configuration.numFrames;
            out.resize(d_configuration.recordLength*d_configuration.bytesPerPoint*d_configuration.numFrames);
        }
        else
            out.resize(d_configuration.recordLength*d_configuration.bytesPerPoint);


        for(int i=0; i<frames; i++)
        {
            for(int j=0; j<d_configuration.recordLength; j++)
            {
                //using the value function here because j could exceed simulated data size
                double dat = d_simulatedData.value(j);


                if(d_configuration.bytesPerPoint == 1)
                {
                    int noise = (rand()%32)-16;
                    qint8 n = qBound(-128,((int)(dat/d_configuration.yMult)+noise),127);
                    out[d_configuration.recordLength*i + j] = n;
                }
                else
                {
                    int noise = (rand()%4096)-2048;
                    qint16 n = qBound(-32768,((int)(dat/d_configuration.yMult)+noise),32767);
                    quint8 byte1;
                    quint8 byte2;
                    if(d_configuration.byteOrder == QDataStream::LittleEndian)
                    {
                        byte1 = (n & 0x00ff);
                        byte2 = (n & 0xff00) >> 8;
                    }
                    else
                    {
                        byte1 = (n & 0xff00) >> 8;
                        byte2 = (n & 0x00ff);
                    }
                    out[d_configuration.recordLength*2*i + 2*j] = byte1;
                    out[d_configuration.recordLength*2*i + 2*j + 1] = byte2;
                }
            }
        }
    //    emit logMessage(QString("Simulate: %1 ms").arg(d_testTime.elapsed()));
        emit shotAcquired(out);
}
