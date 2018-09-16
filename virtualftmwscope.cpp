#include "virtualftmwscope.h"

#include <QFile>
#include <math.h>

VirtualFtmwScope::VirtualFtmwScope(QObject *parent) :
    FtmwScope(parent)
{
    d_subKey = QString("virtual");
    d_prettyName = QString("Virtual FTMW Oscilloscope");
    d_commType = CommunicationProtocol::Virtual;

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(d_key);
    s.beginGroup(d_subKey);
    double bandwidth = s.value(QString("bandwidth"),16000.0).toDouble();
    s.setValue(QString("bandwidth"),bandwidth);

    if(s.beginReadArray(QString("sampleRates")) < 1)
    {
        s.endArray();
        QList<QPair<QString,double>> sampleRates;
        sampleRates << qMakePair(QString("2 GSa/s"),2e9) << qMakePair(QString("5 GSa/s"),5e9)  << qMakePair(QString("10 GSa/s"),10e9)
                    << qMakePair(QString("20 GSa/s"),20e9) << qMakePair(QString("50 GSa/s"),50e9)  << qMakePair(QString("100 GSa/s"),100e9);

        s.beginWriteArray(QString("sampleRates"));
        for(int i=0; i<sampleRates.size(); i++)
        {
            s.setArrayIndex(i);
            s.setValue(QString("text"),sampleRates.at(i).first);
            s.setValue(QString("val"),sampleRates.at(i).second);
        }
    }

    s.endArray();
    s.endGroup();
    s.endGroup();
}

VirtualFtmwScope::~VirtualFtmwScope()
{

}



bool VirtualFtmwScope::testConnection()
{
    d_simulatedTimer->stop();
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    int shotInterval = s.value(QString("%1/%2/shotIntervalMs").arg(key()).arg(subKey()),200).toInt();
    d_simulatedTimer->setInterval(shotInterval);


    emit connected();
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
    testConnection();
}

Experiment VirtualFtmwScope::prepareForExperiment(Experiment exp)
{
    //make a copy of the configuration in which to store settings
    if(!exp.ftmwConfig().isEnabled())
        return exp;

    BlackChirp::FtmwScopeConfig config(exp.ftmwConfig().scopeConfig());

    config.yOff = 0;
    config.yMult = config.vScale*5.0/pow(2.0,8.0*config.bytesPerPoint-1.0);
    config.xIncr = 1.0/config.sampleRate;
    d_configuration = config;
    exp.setScopeConfig(config);
    return exp;

}

void VirtualFtmwScope::beginAcquisition()
{
    d_simulatedTimer->start();
}

void VirtualFtmwScope::endAcquisition()
{
    d_simulatedTimer->stop();
}

void VirtualFtmwScope::readTimeData()
{
}

void VirtualFtmwScope::readWaveform()
{
    //    d_testTime.restart();
        QByteArray out;

        int frames = 1;
        if(d_configuration.fastFrameEnabled && !d_configuration.summaryFrame)
        {
            frames = d_configuration.numFrames;
            out.resize(d_simulatedData.size()*d_configuration.bytesPerPoint*d_configuration.numFrames);
        }
        else
            out.resize(d_simulatedData.size()*d_configuration.bytesPerPoint);


        for(int i=0; i<frames; i++)
        {
            for(int j=0; j<d_simulatedData.size(); j++)
            {
                double dat = d_simulatedData.at(j);


                if(d_configuration.bytesPerPoint == 1)
                {
                    int noise = (rand()%32)-16;
                    qint8 n = qBound(-128,((int)(dat/d_configuration.yMult)+noise),127);
                    out[d_simulatedData.size()*i + j] = n;
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
                    out[d_simulatedData.size()*2*i + 2*j] = byte1;
                    out[d_simulatedData.size()*2*i + 2*j + 1] = byte2;
                }
            }
        }
    //    emit logMessage(QString("Simulate: %1 ms").arg(d_testTime.elapsed()));
        emit shotAcquired(out);
}
