#include "virtuallifscope.h"

#include "virtualinstrument.h"

VirtualLifScope::VirtualLifScope(QObject *parent) :
    LifScope(parent)
{
    d_subKey = QString("virtual");
    d_prettyName = QString("Virtual LIF Oscilloscope");

    d_comm = new VirtualInstrument(d_key,this);
    connect(d_comm,&CommunicationProtocol::hardwareFailure,[=](){ emit hardwareFailure(); });
    connect(d_comm,&CommunicationProtocol::logMessage,this,&VirtualLifScope::logMessage);

    setLifVScale(0.02);
    setRefVScale(0.02);
    setHorizontalConfig(1e9,1000);
}

VirtualLifScope::~VirtualLifScope()
{

}



bool VirtualLifScope::testConnection()
{
    emit connected();
    return true;
}

void VirtualLifScope::initialize()
{
    testConnection();
}

Experiment VirtualLifScope::prepareForExperiment(Experiment exp)
{
    //if(!qFuzzyCompare(d_config.vScale1,exp.lifConfig().scopeConfig().vScale1))
    //setLifVScale(exp.lifConfig().scopeConfig().vScale1);
    //if(!qFuzzyCompare(d_config.vScale2,exp.lifConfig().scopeConfig().vScale2))
    //setRefVScale(exp.lifConfig().scopeConfig().vScale2);
    //if(!qFuzzyCompare(d_config.sampleRate,exp.lifConfig().scopeConfig().sampleRate) || d_config.recordLength != exp.lifConfig().scopeConfig().recordLength)
    //setHorizontalConfig(exp.lifConfig().scopeConfig().sampleRate,exp.lifConfig().scopeConfig().recordLength)
    //exp.setLifScopeConfig(d_config);

    return exp;
}

void VirtualLifScope::beginAcquisition()
{
}

void VirtualLifScope::endAcquisition()
{
}

void VirtualLifScope::readTimeData()
{
}

void VirtualLifScope::setLifVScale(double scale)
{
    d_config.vScale1 = scale;
    d_config.yMult1 = d_config.vScale1*5.0/pow(2.0,8.0*d_config.bytesPerPoint-1.0);
}

void VirtualLifScope::setRefVScale(double scale)
{
    d_config.vScale2 = scale;
    d_config.yMult2 = d_config.vScale2*5.0/pow(2.0,8.0*d_config.bytesPerPoint-1.0);
}

void VirtualLifScope::setHorizontalConfig(double sampleRate, int recLen)
{
    d_config.sampleRate = sampleRate;
    d_config.recordLength = recLen;
    d_config.xIncr = 1.0/sampleRate;
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
            if(d_config.byteOrder == QDataStream::LittleEndian)
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
                if(d_config.byteOrder == QDataStream::LittleEndian)
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

    emit waveformRead(d_config,out);

}

void VirtualLifScope::setRefEnabled(bool en)
{
    d_refEnabled = en;

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.setValue(QString("%1/refEnabled"),en);
}
