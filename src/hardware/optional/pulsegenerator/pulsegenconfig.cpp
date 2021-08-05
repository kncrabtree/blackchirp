#include <hardware/optional/pulsegenerator/pulsegenconfig.h>

#include <QMetaEnum>

PulseGenConfig::PulseGenConfig() : HeaderStorage(BC::Store::PGenConfig::key)
{
}

PulseGenConfig::~PulseGenConfig()
{
}

PulseGenConfig::ChannelConfig PulseGenConfig::at(const int i) const
{
    return d_config.value(i);
}

int PulseGenConfig::size() const
{
    return d_config.size();
}

bool PulseGenConfig::isEmpty() const
{
    return d_config.isEmpty();
}

QVariant PulseGenConfig::setting(const int index, const Setting s) const
{
    if(index < 0 || index >= d_config.size())
        return QVariant();

    switch(s)
    {
    case DelaySetting:
        return d_config.at(index).delay;
        break;
    case WidthSetting:
        return d_config.at(index).width;
        break;
    case EnabledSetting:
        return d_config.at(index).enabled;
        break;
    case LevelSetting:
        return d_config.at(index).level;
        break;
    case NameSetting:
        return d_config.at(index).channelName;
        break;
    case RoleSetting:
        return d_config.at(index).role;
        break;
    default:
        break;
    }

    return QVariant();
}

QVector<QVariant> PulseGenConfig::setting(Role role, const Setting s) const
{
    QVector<QVariant> out;
    for(int i=0; i<d_config.size(); i++)
    {
        if(d_config.at(i).role == role)
            out << setting(i,s);
    }

    return out;
}

PulseGenConfig::ChannelConfig PulseGenConfig::settings(const int index) const
{
    if(index < 0 || index >= d_config.size())
        return ChannelConfig();

    return d_config.at(index);

}

QVector<int> PulseGenConfig::channelsForRole(Role role) const
{
    QVector<int> out;
    for(int i=0; i<d_config.size(); i++)
    {
        if(d_config.at(i).role == role)
            out << i;
    }

    return out;
}

double PulseGenConfig::repRate() const
{
    return d_repRate;
}

void PulseGenConfig::set(const int index, const Setting s, const QVariant val)
{
    if(index < 0 || index >= d_config.size())
        return;

    switch(s)
    {
    case DelaySetting:
        d_config[index].delay = val.toDouble();
        break;
    case WidthSetting:
        d_config[index].width = val.toDouble();
        break;
    case EnabledSetting:
        d_config[index].enabled = val.toBool();
        break;
    case LevelSetting:
        d_config[index].level = val.value<ActiveLevel>();
        break;
    case NameSetting:
        if(d_config.at(index).role == None)
            d_config[index].channelName = val.toString();
        else
            d_config[index].channelName = QString(QMetaEnum::fromType<Role>().key(d_config.at(index).role));
        break;
    case RoleSetting:
        d_config[index].role = val.value<Role>();
        if(d_config.at(index).role != None)
            d_config[index].channelName = QString(QMetaEnum::fromType<Role>().key(d_config.at(index).role));
    default:
        break;
    }
}

void PulseGenConfig::set(const int index, const ChannelConfig cc)
{
    if(index < 0 || index >= d_config.size())
        return;

    set(index,DelaySetting,cc.delay);
    set(index,WidthSetting,cc.width);
    set(index,EnabledSetting,cc.enabled);
    set(index,LevelSetting,cc.level);
    set(index,NameSetting,cc.channelName);
    set(index,RoleSetting,cc.role);
}

void PulseGenConfig::set(Role role, const Setting s, const QVariant val)
{
    for(int i=0; i<d_config.size(); i++)
    {
        if(d_config.at(i).role == role)
            set(i,s,val);
    }
}

void PulseGenConfig::set(Role role, const ChannelConfig cc)
{
    for(int i=0; i<d_config.size(); i++)
    {
        if(d_config.at(i).role == role)
            set(i,cc);
    }
}

void PulseGenConfig::addChannel()
{
    d_config.append(ChannelConfig());
}

void PulseGenConfig::setRepRate(const double r)
{
    d_repRate = r;
}



void PulseGenConfig::storeValues()
{
    using namespace BC::Store::PGenConfig;
    store(rate,d_repRate,QString("Hz"));
    for(int i=0; i<d_config.size(); ++i)
    {
        auto &cc = d_config.at(i);
        storeArrayValue(channel,i,name,cc.channelName);
        storeArrayValue(channel,i,delay,cc.delay,QString::fromUtf8("μs"));
        storeArrayValue(channel,i,width,cc.width,QString::fromUtf8("μs"));
        storeArrayValue(channel,i,level,cc.level);
        storeArrayValue(channel,i,enabled,cc.enabled);
        storeArrayValue(channel,i,role,cc.role);
    }
}

void PulseGenConfig::retrieveValues()
{
    using namespace BC::Store::PGenConfig;
    d_repRate = retrieve(rate,0.0);
    auto n = arrayStoreSize(channel);
    d_config.clear();
    d_config.reserve(n);
    for(std::size_t i =0; i<n; ++i)
    {
        ChannelConfig cc {
                    retrieveArrayValue(channel,i,name,QString("")),
                    retrieveArrayValue(channel,i,enabled,false),
                    retrieveArrayValue(channel,i,delay,0.0),
                    retrieveArrayValue(channel,i,width,1.0),
                    retrieveArrayValue(channel,i,level,ActiveHigh),
                    retrieveArrayValue(channel,i,role,None)
        };

        d_config << cc;
    }
}
