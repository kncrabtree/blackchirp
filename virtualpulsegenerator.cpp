#include "virtualpulsegenerator.h"

#include "virtualinstrument.h"

VirtualPulseGenerator::VirtualPulseGenerator(QObject *parent) : PulseGenerator(parent)
{
    d_subKey = QString("virtual");
    d_prettyName = QString("Virtual Pulse Generator");

    d_comm = new VirtualInstrument(d_key,this);
    connect(d_comm,&CommunicationProtocol::logMessage,this,&VirtualPulseGenerator::logMessage);
    connect(d_comm,&CommunicationProtocol::hardwareFailure,[=](){ emit hardwareFailure(); });
}

VirtualPulseGenerator::~VirtualPulseGenerator()
{

}



bool VirtualPulseGenerator::testConnection()
{
    blockSignals(true);
    readAll();
    blockSignals(false);

    emit configUpdate(d_config);
    emit connected();
    return true;
}

void VirtualPulseGenerator::initialize()
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(d_key);

    s.beginReadArray(QString("channels"));
    for(int i=0; i<BC_PGEN_NUMCHANNELS; i++)
    {
        s.setArrayIndex(i);
        QString name = s.value(QString("name"),QString("Ch%1").arg(i)).toString();
        double d = s.value(QString("defaultDelay"),0.0).toDouble();
        double w = s.value(QString("defaultWidth"),0.050).toDouble();
        QVariant lvl = s.value(QString("level"),BlackChirp::PulseLevelActiveHigh);
        bool en = s.value(QString("defaultEnabled"),false).toBool();

        if(lvl == QVariant(BlackChirp::PulseLevelActiveHigh))
            d_config.add(name,en,d,w,BlackChirp::PulseLevelActiveHigh);
        else
            d_config.add(name,en,d,w,BlackChirp::PulseLevelActiveLow);
    }
    s.endArray();

    d_config.setRepRate(s.value(QString("repRate"),10.0).toDouble());
    s.endGroup();

    testConnection();
}

Experiment VirtualPulseGenerator::prepareForExperiment(Experiment exp)
{
    setAll(exp.pGenConfig());
    return exp;
}

void VirtualPulseGenerator::beginAcquisition()
{
}

void VirtualPulseGenerator::endAcquisition()
{
}

void VirtualPulseGenerator::readTimeData()
{
}

QVariant VirtualPulseGenerator::read(const int index, const BlackChirp::PulseSetting s)
{
    emit settingUpdate(index,s,d_config.setting(index,s));
    return d_config.setting(index,s);
}

QVariant VirtualPulseGenerator::set(const int index, const BlackChirp::PulseSetting s, const QVariant val)
{
    d_config.set(index,s,val);
    return read(index,s);
}

void VirtualPulseGenerator::setRepRate(double d)
{
    d_config.setRepRate(d);
    emit repRateUpdate(d);
}

double VirtualPulseGenerator::setLifDelay(double d)
{
   return set(BC_PGEN_LIFCHANNEL,BlackChirp::PulseDelay,d).toDouble();
}
