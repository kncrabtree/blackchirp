#include "pulsegenerator.h"

PulseGenerator::PulseGenerator(QObject *parent) :
    Rs232Instrument(QString("pGen"),QString("Pulse Generator"),parent), d_numChannels(8)
{
#ifdef BC_NOPULSEGEN
    d_virtual = true;
#endif
}

PulseGenerator::~PulseGenerator()
{

}

PulseGenConfig PulseGenerator::config() const
{
    return d_config;
}



bool PulseGenerator::testConnection()
{
    if(!Rs232Instrument::testConnection())
    {
        emit connectionResult(this,false,QString("RS232 error."));
        return false;
    }

    if(!d_virtual)
    {

    }

    readAll();
    emit connectionResult(this,true);
    return true;
}

void PulseGenerator::initialize()
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(d_key);

    d_config.add(QString("Gas"),true,0.0,400.0,
                 s.value(QString("ch0Level"),PulseGenConfig::ActiveHigh).value<PulseGenConfig::ActiveLevel>());
    d_config.add(QString("AWG"),false,500.0,1.0,
                 s.value(QString("ch1Level"),PulseGenConfig::ActiveHigh).value<PulseGenConfig::ActiveLevel>());
    d_config.add(QString("Xmer"),false,515.0,1.0,
                 s.value(QString("ch2Level"),PulseGenConfig::ActiveHigh).value<PulseGenConfig::ActiveLevel>());
    d_config.add(QString("LIF"),false,530.0,1.0,
                 s.value(QString("ch3Level"),PulseGenConfig::ActiveHigh).value<PulseGenConfig::ActiveLevel>());
    d_config.add(s.value(QString("ch4Name"),QString("Aux1")).toString(),false,0.0,50.0,
                 s.value(QString("ch4Level"),PulseGenConfig::ActiveHigh).value<PulseGenConfig::ActiveLevel>());
    d_config.add(s.value(QString("ch5Name"),QString("Aux2")).toString(),false,0.0,50.0,
                 s.value(QString("ch5Level"),PulseGenConfig::ActiveHigh).value<PulseGenConfig::ActiveLevel>());
    d_config.add(s.value(QString("ch6Name"),QString("Aux3")).toString(),false,0.0,50.0,
                 s.value(QString("ch6Level"),PulseGenConfig::ActiveHigh).value<PulseGenConfig::ActiveLevel>());
    d_config.add(s.value(QString("ch7Name"),QString("Aux4")).toString(),false,0.0,50.0,
                 s.value(QString("ch7Level"),PulseGenConfig::ActiveHigh).value<PulseGenConfig::ActiveLevel>());

    s.endGroup();

    Rs232Instrument::initialize();
    testConnection();
}

Experiment PulseGenerator::prepareForExperiment(Experiment exp)
{
    return exp;
}

void PulseGenerator::beginAcquisition()
{
}

void PulseGenerator::endAcquisition()
{
}

void PulseGenerator::readTimeData()
{
}

QVariant PulseGenerator::read(const int index, const PulseGenConfig::Setting s)
{
    if(d_virtual)
    {
        emit settingUpdate(index,s,d_config.setting(index,s));
        return d_config.setting(index,s);
    }

    //implement communication
    return QVariant();
}

PulseGenConfig::ChannelConfig PulseGenerator::read(const int index)
{
    PulseGenConfig::ChannelConfig out = d_config.settings(index);
    bool ok = false;
    out.channelName = read(index,PulseGenConfig::Name).toString();
    out.delay = read(index,PulseGenConfig::Delay).toDouble(&ok);
    if(!ok)
        return out;
    out.width = read(index,PulseGenConfig::Width).toDouble(&ok);
    if(!ok)
        return out;
    out.enabled = read(index,PulseGenConfig::Enabled).toBool();
    out.level = read(index,PulseGenConfig::Level).value<PulseGenConfig::ActiveLevel>();

    return out;
}

void PulseGenerator::set(const int index, const PulseGenConfig::Setting s, const QVariant val)
{
    if(d_virtual)
    {
        d_config.set(index,s,val);
        return;
    }

    //communicate
}

void PulseGenerator::setChannel(const int index, const PulseGenConfig::ChannelConfig cc)
{
    set(index,PulseGenConfig::Name,cc.channelName);
    set(index,PulseGenConfig::Enabled,cc.enabled);
    set(index,PulseGenConfig::Delay,cc.delay);
    set(index,PulseGenConfig::Width,cc.width);
    set(index,PulseGenConfig::Level,cc.level);
}

void PulseGenerator::setAll(const PulseGenConfig cc)
{
    if(d_virtual)
    {
        d_config = cc;
        return;
    }

    for(int i=0; i<d_config.size(); i++)
        setChannel(i,cc.at(i));

    return;
}

void PulseGenerator::readAll()
{
    for(int i=0;i<d_numChannels; i++)
        read(i);

}
