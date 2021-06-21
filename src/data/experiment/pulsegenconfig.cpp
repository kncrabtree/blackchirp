#include <src/data/experiment/pulsegenconfig.h>



PulseGenConfig::PulseGenConfig() : data(new PulseGenConfigData)
{

}

PulseGenConfig::PulseGenConfig(const PulseGenConfig &rhs) : data(rhs.data)
{

}

PulseGenConfig &PulseGenConfig::operator=(const PulseGenConfig &rhs)
{
    if (this != &rhs)
        data.operator=(rhs.data);
    return *this;
}

PulseGenConfig::~PulseGenConfig()
{

}

PulseGenConfig::ChannelConfig PulseGenConfig::at(const int i) const
{
    Q_ASSERT(i < data->config.size());
    return data->config.at(i);
}

int PulseGenConfig::size() const
{
    return data->config.size();
}

bool PulseGenConfig::isEmpty() const
{
    return data->config.isEmpty();
}

QVariant PulseGenConfig::setting(const int index, const Setting s) const
{
    if(index < 0 || index >= data->config.size())
        return QVariant();

    switch(s)
    {
    case DelaySetting:
        return data->config.at(index).delay;
        break;
    case WidthSetting:
        return data->config.at(index).width;
        break;
    case EnabledSetting:
        return data->config.at(index).enabled;
        break;
    case LevelSetting:
        return data->config.at(index).level;
        break;
    case NameSetting:
        return data->config.at(index).channelName;
        break;
    case RoleSetting:
        return data->config.at(index).role;
        break;
    default:
        break;
    }

    return QVariant();
}

QList<QVariant> PulseGenConfig::setting(Role role, const Setting s) const
{
    QList<QVariant> out;
    for(int i=0; i<data->config.size(); i++)
    {
        if(data->config.at(i).role == role)
            out << setting(i,s);
    }

    return out;
}

PulseGenConfig::ChannelConfig PulseGenConfig::settings(const int index) const
{
    if(index < 0 || index >= data->config.size())
        return ChannelConfig();

    return data->config.at(index);

}

QList<int> PulseGenConfig::channelsForRole(Role role) const
{
    QList<int> out;
    for(int i=0; i<data->config.size(); i++)
    {
        if(data->config.at(i).role == role)
            out << i;
    }

    return out;
}

double PulseGenConfig::repRate() const
{
    return data->repRate;
}

QMap<QString, QPair<QVariant, QString> > PulseGenConfig::headerMap() const
{
    QMap<QString, QPair<QVariant, QString> > out;

    for(int i=0; i<data->config.size(); i++)
    {
        out.insert(QString("PulseGen.%1.Name").arg(i),qMakePair(data->config.at(i).channelName,QString("")));
        out.insert(QString("PulseGen.%1.Enabled").arg(i),qMakePair(data->config.at(i).enabled,QString("")));
        out.insert(QString("PulseGen.%1.Delay").arg(i),qMakePair(QString::number(data->config.at(i).delay,'f',3),QString::fromUtf16(u"µs")));
        out.insert(QString("PulseGen.%1.Width").arg(i),qMakePair(QString::number(data->config.at(i).width,'f',3),QString::fromUtf16(u"µs")));
        out.insert(QString("PulseGen.%1.Level").arg(i),qMakePair(data->config.at(i).level,QString("")));
        out.insert(QString("PulseGen.%1.Role").arg(i),qMakePair(data->config.at(i).role,roles.value(data->config.at(i).role)));
    }
    out.insert(QString("PulseGenRepRate"),qMakePair(QString::number(data->repRate,'f',1),QString("Hz")));

    return out;
}

void PulseGenConfig::parseLine(QString key, QVariant val)
{
    if(key.startsWith(QString("PulseGen")))
    {
        if(key.endsWith(QString("RepRate")))
            data->repRate = val.toDouble();
        else
        {
            QStringList l = key.split(QString("."));
            if(l.size() < 3)
                return;

            QString subKey = l.constLast();
            int index = l.at(1).toInt();

            while(data->config.size() <= index)
            {
                ChannelConfig c;
                data->config.append(c);
            }

            if(subKey.endsWith(QString("Name")))
            {
                if(data->config.at(index).role == NoRole)
                    data->config[index].channelName = val.toString();
            }
            if(subKey.endsWith(QString("Enabled")))
                data->config[index].enabled = val.toBool();
            if(subKey.endsWith(QString("Delay")))
                data->config[index].delay = val.toDouble();
            if(subKey.endsWith(QString("Width")))
                data->config[index].width = val.toDouble();
            if(subKey.endsWith(QString("Level")))
                data->config[index].level = (ActiveLevel)val.toInt();
            if(subKey.endsWith(QString("Role")))
            {
                data->config[index].role = static_cast<Role>(val.toInt());
                data->config[index].channelName = roles.value(data->config.at(index).role);
            }
        }
    }
}

void PulseGenConfig::set(const int index, const Setting s, const QVariant val)
{
    if(index < 0 || index >= data->config.size())
        return;

    switch(s)
    {
    case DelaySetting:
        data->config[index].delay = val.toDouble();
        break;
    case WidthSetting:
        data->config[index].width = val.toDouble();
        break;
    case EnabledSetting:
        data->config[index].enabled = val.toBool();
        break;
    case LevelSetting:
        data->config[index].level = val.value<ActiveLevel>();
        break;
    case NameSetting:
        if(data->config.at(index).role == NoRole)
            data->config[index].channelName = val.toString();
        else
            data->config[index].channelName = roles.value(data->config.at(index).role);
        break;
    case RoleSetting:
        data->config[index].role = val.value<Role>();
        if(data->config.at(index).role != NoRole)
            data->config[index].channelName = roles.value(data->config.at(index).role);
    default:
        break;
    }
}

void PulseGenConfig::set(const int index, const ChannelConfig cc)
{
    if(index < 0 || index >= data->config.size())
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
    for(int i=0; i<data->config.size(); i++)
    {
        if(data->config.at(i).role == role)
            set(i,s,val);
    }
}

void PulseGenConfig::set(Role role, const ChannelConfig cc)
{
    for(int i=0; i<data->config.size(); i++)
    {
        if(data->config.at(i).role == role)
            set(i,cc);
    }
}

void PulseGenConfig::addChannel()
{
    data->config.append(ChannelConfig());
}

void PulseGenConfig::setRepRate(const double r)
{
    data->repRate = r;
}

