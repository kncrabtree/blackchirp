#include "pulsegenerator.h"

PulseGenerator::PulseGenerator(QObject *parent) :
    Rs232Instrument(QString("pGen"),QString("Pulse Generator"),parent)
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

    blockSignals(true);
    readAll();
    blockSignals(false);

    emit configUpdate(d_config);
    emit connectionResult(this,true);
    return true;
}

void PulseGenerator::initialize()
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
        QVariant lvl = s.value(QString("level"),PulseGenConfig::ActiveHigh);
        bool en = s.value(QString("defaultEnabled"),false).toBool();

        if(lvl == QVariant(PulseGenConfig::ActiveHigh))
            d_config.add(name,en,d,w,PulseGenConfig::ActiveHigh);
        else
            d_config.add(name,en,d,w,PulseGenConfig::ActiveLow);
    }
    s.endArray();

    d_config.setRepRate(s.value(QString("repRate"),10.0).toDouble());
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
        emit settingUpdate(index,s,val);
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

    setRepRate(cc.repRate());

    return;
}

void PulseGenerator::setRepRate(double d)
{
    if(d_virtual)
    {
        d_config.setRepRate(d);
        emit repRateUpdate(d);
        return;
    }

    //communicate
}

void PulseGenerator::readAll()
{
    for(int i=0;i<BC_PGEN_NUMCHANNELS; i++)
        read(i);

}
