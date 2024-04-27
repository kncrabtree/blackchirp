#include <hardware/optional/pulsegenerator/pulsegenconfig.h>

#include <QMetaEnum>
#include <hardware/optional/pulsegenerator/pulsegenerator.h>

PulseGenConfig::PulseGenConfig(QString subKey, int index) : HeaderStorage(BC::Key::hwKey(BC::Key::PGen::key,index),subKey)
{
}

PulseGenConfig::~PulseGenConfig()
{
}

PulseGenConfig::ChannelConfig PulseGenConfig::at(const int i) const
{
    return d_channels.value(i);
}

int PulseGenConfig::size() const
{
    return d_channels.size();
}

bool PulseGenConfig::isEmpty() const
{
    return d_channels.isEmpty();
}

QVariant PulseGenConfig::setting(const int index, const Setting s) const
{
    if(index >= d_channels.size())
        return QVariant();

    switch(s)
    {
    case DelaySetting:
        return d_channels.at(index).delay;
        break;
    case WidthSetting:
        return d_channels.at(index).width;
        break;
    case EnabledSetting:
        return d_channels.at(index).enabled;
        break;
    case LevelSetting:
        return d_channels.at(index).level;
        break;
    case NameSetting:
        return d_channels.at(index).channelName;
        break;
    case RoleSetting:
        return d_channels.at(index).role;
        break;
    case ModeSetting:
        return d_channels.at(index).mode;
        break;
    case SyncSetting:
        return d_channels.at(index).syncCh;
        break;
    case DutyOnSetting:
        return d_channels.at(index).dutyOn;
        break;
    case DutyOffSetting:
        return d_channels.at(index).dutyOff;
        break;
    case RepRateSetting:
        return d_repRate;
        break;
    case PGenModeSetting:
        return d_mode;
        break;
    case PGenEnabledSetting:
        return d_pulseEnabled;
        break;
    }

    return QVariant();
}

QVariant PulseGenConfig::setting(Role role, const Setting s) const
{
    for(int i=0; i<d_channels.size(); i++)
    {
        if(d_channels.at(i).role == role)
            return setting(i,s);
    }

    return QVariant();
}

PulseGenConfig::ChannelConfig PulseGenConfig::settings(const int index) const
{
    if(index < 0 || index >= d_channels.size())
        return ChannelConfig();

    return d_channels.at(index);

}

QVector<PulseGenConfig::Role> PulseGenConfig::activeRoles() const
{
    QVector<PulseGenConfig::Role> out;
    for(int i=0; i<d_channels.size(); i++)
    {
        if(d_channels.at(i).role != None)
            out.append(d_channels.at(i).role);
    }
    return out;
}

QVector<int> PulseGenConfig::channelsForRole(Role role) const
{
    QVector<int> out;
    for(int i=0; i<d_channels.size(); i++)
    {
        if(d_channels.at(i).role == role)
            out << i;
    }

    return out;
}

double PulseGenConfig::channelStart(const int index) const
{
    if(index < 0 || index >= d_channels.size())
        return 0.0;

    auto ch = d_channels.at(index);
    if(ch.syncCh == 0)
        return ch.delay;
    else
        return ch.delay + channelStart(ch.syncCh - 1);
}

bool PulseGenConfig::testCircularSync(const int index, int newSyncCh)
{
    auto sc = newSyncCh;
    while(sc != 0)
    {
        sc = d_channels.at(sc-1).syncCh;
        if(sc == index+1)
            return true;
    }

    return false;
}

void PulseGenConfig::setCh(const int index, const Setting s, const QVariant val)
{
    if(index < 0 || index >= d_channels.size())
        return;

    switch(s)
    {
    case DelaySetting:
        d_channels[index].delay = val.toDouble();
        break;
    case WidthSetting:
        d_channels[index].width = val.toDouble();
        break;
    case EnabledSetting:
        d_channels[index].enabled = val.toBool();
        break;
    case LevelSetting:
        d_channels[index].level = val.value<ActiveLevel>();
        break;
    case NameSetting:
        if(d_channels.at(index).role == None)
            d_channels[index].channelName = val.toString();
        else
            d_channels[index].channelName = QString(QMetaEnum::fromType<Role>().valueToKey(d_channels.at(index).role));
        break;
    case RoleSetting:
        d_channels[index].role = val.value<Role>();
        if(d_channels.at(index).role != None)
            d_channels[index].channelName = QString(QMetaEnum::fromType<Role>().valueToKey(d_channels.at(index).role));
        break;
    case ModeSetting:
        d_channels[index].mode = val.value<ChannelMode>();
        break;
    case SyncSetting:
        d_channels[index].syncCh = val.toInt();
        break;
    case DutyOnSetting:
        d_channels[index].dutyOn = val.toInt();
        break;
    case DutyOffSetting:
        d_channels[index].dutyOff = val.toInt();
        break;
    case RepRateSetting:
        d_repRate = val.toDouble();
        break;
    case PGenModeSetting:
        d_mode = val.value<PGenMode>();
        break;
    case PGenEnabledSetting:
        d_pulseEnabled = val.toBool();
        break;
    }
}

void PulseGenConfig::setCh(const int index, const ChannelConfig cc)
{
    if(index < 0 || index >= d_channels.size())
        return;

    setCh(index,DelaySetting,cc.delay);
    setCh(index,WidthSetting,cc.width);
    setCh(index,EnabledSetting,cc.enabled);
    setCh(index,LevelSetting,cc.level);
    setCh(index,NameSetting,cc.channelName);
    setCh(index,RoleSetting,cc.role);
    setCh(index,ModeSetting,cc.mode);
    setCh(index,SyncSetting,cc.syncCh);
    setCh(index,DutyOnSetting,cc.dutyOn);
    setCh(index,DutyOffSetting,cc.dutyOff);
}

void PulseGenConfig::setCh(Role role, const Setting s, const QVariant val)
{
    for(int i=0; i<d_channels.size(); i++)
    {
        if(d_channels.at(i).role == role)
        {
            setCh(i,s,val);
            return;
        }
    }
}

void PulseGenConfig::setCh(Role role, const ChannelConfig cc)
{
    for(int i=0; i<d_channels.size(); i++)
    {
        if(d_channels.at(i).role == role)
        {
            setCh(i,cc);
            return;
        }
    }
}

void PulseGenConfig::addChannel()
{
    d_channels.append(ChannelConfig());
}



void PulseGenConfig::storeValues()
{
    using namespace BC::Store::PGenConfig;
    store(rate,d_repRate,BC::Unit::Hz);
    store(pGenMode,d_mode);
    store(pGenEnabled,d_pulseEnabled);
    for(int i=0; i<d_channels.size(); ++i)
    {
        auto &cc = d_channels.at(i);
        storeArrayValue(channel,i,name,cc.channelName);
        storeArrayValue(channel,i,delay,cc.delay,BC::Unit::us);
        storeArrayValue(channel,i,width,cc.width,BC::Unit::us);
        storeArrayValue(channel,i,level,cc.level);
        storeArrayValue(channel,i,enabled,cc.enabled);
        storeArrayValue(channel,i,role,cc.role);
        storeArrayValue(channel,i,chMode,cc.mode);
        storeArrayValue(channel,i,sync,cc.syncCh);
        storeArrayValue(channel,i,dutyOn,cc.dutyOn);
        storeArrayValue(channel,i,dutyOff,cc.dutyOff);
    }
}

void PulseGenConfig::retrieveValues()
{
    using namespace BC::Store::PGenConfig;
    d_repRate = retrieve(rate,0.0);
    d_pulseEnabled = retrieve(pGenEnabled,true);
    d_mode = retrieve(pGenMode,Continuous);
    auto n = arrayStoreSize(channel);
    d_channels.clear();
    d_channels.reserve(n);
    for(std::size_t i =0; i<n; ++i)
    {
        ChannelConfig cc {
                    retrieveArrayValue(channel,i,name,QString("")),
                    retrieveArrayValue(channel,i,enabled,false),
                    retrieveArrayValue(channel,i,delay,0.0),
                    retrieveArrayValue(channel,i,width,1.0),
                    retrieveArrayValue(channel,i,level,ActiveHigh),
                    retrieveArrayValue(channel,i,role,None),
                    retrieveArrayValue(channel,i,chMode,Normal),
                    retrieveArrayValue(channel,i,sync,0),
                    retrieveArrayValue(channel,i,dutyOn,1),
                    retrieveArrayValue(channel,i,dutyOff,1)
        };

        d_channels << cc;
    }
}
