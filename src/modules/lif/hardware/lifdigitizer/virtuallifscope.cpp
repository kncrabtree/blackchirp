#include "virtuallifscope.h"

#include <QTimer>
#include <math.h>

VirtualLifScope::VirtualLifScope(QObject *parent) :
    LifScope(BC::Key::hwVirtual,BC::Key::vLifScopeName,CommunicationProtocol::Virtual,parent)
{
    setLifVScale(0.02);
    setRefVScale(0.02);
    setHorizontalConfig(1e9,1000);
}

VirtualLifScope::~VirtualLifScope()
{

}

void VirtualLifScope::readSettings()
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(d_key);
    s.beginGroup(d_subKey);
    d_config.refEnabled = s.value(QString("refEnabled"),false).toBool();
    double minVS = s.value(QString("minVScale"),0.01).toDouble();
    double maxVS = s.value(QString("maxVScale"),5.0).toDouble();
    double minSamples = s.value(QString("minSamples"),1000).toInt();
    double maxSamples = s.value(QString("maxSamples"),10000).toInt();
    s.setValue(QString("minVScale"),minVS);
    s.setValue(QString("maxVScale"),maxVS);
    s.setValue(QString("minSamples"),minSamples);
    s.setValue(QString("maxSamples"),maxSamples);
    s.setValue(QString("refEnabled"),d_config.refEnabled);
    s.endGroup();
    s.endGroup();
    s.sync();

    setRefEnabled(d_config.refEnabled);
}



bool VirtualLifScope::testConnection()
{
    return true;
}

void VirtualLifScope::initialize()
{

    p_timer = new QTimer(this);
    p_timer->setInterval(200);
    connect(p_timer,&QTimer::timeout,this,&VirtualLifScope::queryScope);
    p_timer->start();
}

void VirtualLifScope::setLifVScale(double scale)
{
    d_config.vScale1 = scale;
    d_config.yMult1 = d_config.vScale1*5.0/pow(2.0,8.0*d_config.bytesPerPoint-1.0);
    emit configUpdated(d_config);
}

void VirtualLifScope::setRefVScale(double scale)
{
    d_config.vScale2 = scale;
    d_config.yMult2 = d_config.vScale2*5.0/pow(2.0,8.0*d_config.bytesPerPoint-1.0);
    emit configUpdated(d_config);
}

void VirtualLifScope::setHorizontalConfig(double sampleRate, int recLen)
{
    d_config.sampleRate = sampleRate;
    d_config.recordLength = recLen;
    d_config.xIncr = 1.0/sampleRate;
    emit configUpdated(d_config);
}

void VirtualLifScope::queryScope()
{
    QByteArray out;
    if(d_config.refEnabled)
        out.resize(2*d_config.recordLength*d_config.bytesPerPoint);
    else
        out.resize(d_config.recordLength*d_config.bytesPerPoint);

    for(int i=0; i<d_config.recordLength; i++)
    {
        if(d_config.bytesPerPoint == 1)
        {
            qint8 dat = (qrand() % 256) - 128;
            out[i] = dat;
        }
        else
        {
            qint16 dat = (qrand() % 65536) - 32768;
            qint8 datmsb = dat / 256;
            qint8 datlsb = dat % 256;
            if(d_config.byteOrder == DigitizerConfig::LittleEndian)
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

    if(d_config.refEnabled)
    {
        for(int i = d_config.recordLength; i<2*d_config.recordLength; i++)
        {
            if(d_config.bytesPerPoint == 1)
            {
                qint8 dat = (qrand() % 256) - 128;
                out[i] = dat;
            }
            else
            {
                qint16 dat = (qrand() % 65536) - 32768;
                qint8 datmsb = dat / 256;
                qint8 datlsb = dat % 256;
                if(d_config.byteOrder == DigitizerConfig::LittleEndian)
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

    emit waveformRead(LifTrace(d_config,out));

}

void VirtualLifScope::setRefEnabled(bool en)
{
    d_config.refEnabled = en;

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(d_key);
    s.beginGroup(d_subKey);
    s.setValue(QString("refEnabled"),en);
    s.endGroup();
    s.endGroup();
    s.sync();

    emit configUpdated(d_config);
}


void VirtualLifScope::sleep(bool b)
{
    if(b)
        p_timer->stop();
    else
    {
        p_timer->start();
    }
}
