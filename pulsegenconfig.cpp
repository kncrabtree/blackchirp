#include "pulsegenconfig.h"



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

BlackChirp::PulseChannelConfig PulseGenConfig::at(const int i) const
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

QVariant PulseGenConfig::setting(const int index, const BlackChirp::PulseSetting s) const
{
    if(index < 0 || index >= data->config.size())
        return QVariant();

    switch(s)
    {
    case BlackChirp::PulseDelaySetting:
        return data->config.at(index).delay;
        break;
    case BlackChirp::PulseWidthSetting:
        return data->config.at(index).width;
        break;
    case BlackChirp::PulseEnabledSetting:
        return data->config.at(index).enabled;
        break;
    case BlackChirp::PulseLevelSetting:
        return data->config.at(index).level;
        break;
    case BlackChirp::PulseNameSetting:
        return data->config.at(index).channelName;
        break;
    case BlackChirp::PulseRoleSetting:
        return data->config.at(index).role;
        break;
    default:
        break;
    }

    return QVariant();
}

BlackChirp::PulseChannelConfig PulseGenConfig::settings(const int index) const
{
    if(index < 0 || index >= data->config.size())
        return BlackChirp::PulseChannelConfig();

    return data->config.at(index);

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
        out.insert(QString("PulseGen.%1.Role").arg(i),qMakePair(data->config.at(i).role,BlackChirp::getPulseName(data->config.at(i).role)));
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
                BlackChirp::PulseChannelConfig c;
                c.channel = data->config.size()+1;
                data->config.append(c);
            }

            if(subKey.endsWith(QString("Name")))
            {
                if(data->config.at(index).role == BlackChirp::NoPulseRole)
                    data->config[index].channelName = val.toString();
            }
            if(subKey.endsWith(QString("Enabled")))
                data->config[index].enabled = val.toBool();
            if(subKey.endsWith(QString("Delay")))
                data->config[index].delay = val.toDouble();
            if(subKey.endsWith(QString("Width")))
                data->config[index].width = val.toDouble();
            if(subKey.endsWith(QString("Level")))
                data->config[index].level = (BlackChirp::PulseActiveLevel)val.toInt();
            if(subKey.endsWith(QString("Role")))
            {
                data->config[index].role = static_cast<BlackChirp::PulseRole>(val.toInt());
                data->config[index].channelName = BlackChirp::getPulseName(data->config.at(index).role);
            }
        }
    }
}

void PulseGenConfig::set(const int index, const BlackChirp::PulseSetting s, const QVariant val)
{
    if(index < 0 || index >= data->config.size())
        return;

    switch(s)
    {
    case BlackChirp::PulseDelaySetting:
        data->config[index].delay = val.toDouble();
        break;
    case BlackChirp::PulseWidthSetting:
        data->config[index].width = val.toDouble();
        break;
    case BlackChirp::PulseEnabledSetting:
        data->config[index].enabled = val.toBool();
        break;
    case BlackChirp::PulseLevelSetting:
        data->config[index].level = static_cast<BlackChirp::PulseActiveLevel>(val.toInt());
        break;
    case BlackChirp::PulseNameSetting:
        if(data->config.at(index).role == BlackChirp::NoPulseRole)
            data->config[index].channelName = val.toString();
        else
            data->config[index].channelName = BlackChirp::getPulseName(data->config.at(index).role);
        break;
    case BlackChirp::PulseRoleSetting:
        data->config[index].role = static_cast<BlackChirp::PulseRole>(val.toInt());
        if(data->config.at(index).role != BlackChirp::NoPulseRole)
            data->config[index].channelName = BlackChirp::getPulseName(data->config.at(index).role);
    default:
        break;
    }
}

void PulseGenConfig::set(const int index, const BlackChirp::PulseChannelConfig cc)
{
    if(index < 0 || index >= data->config.size())
        return;

    set(index,BlackChirp::PulseDelaySetting,cc.delay);
    set(index,BlackChirp::PulseWidthSetting,cc.width);
    set(index,BlackChirp::PulseEnabledSetting,cc.enabled);
    set(index,BlackChirp::PulseLevelSetting,cc.level);
    set(index,BlackChirp::PulseNameSetting,cc.channelName);
    set(index,BlackChirp::PulseRoleSetting,cc.role);
}

void PulseGenConfig::set(BlackChirp::PulseRole role, const BlackChirp::PulseSetting s, const QVariant val)
{
    for(int i=0; i<data->config.size(); i++)
    {
        if(data->config.at(i).role == role)
            set(i,s,val);
    }
}

void PulseGenConfig::set(BlackChirp::PulseRole role, const BlackChirp::PulseChannelConfig cc)
{
    for(int i=0; i<data->config.size(); i++)
    {
        if(data->config.at(i).role == role)
            set(i,cc);
    }
}

void PulseGenConfig::add(const QString name, const bool enabled, const double delay, const double width, const BlackChirp::PulseActiveLevel level, const BlackChirp::PulseRole role)
{
    BlackChirp::PulseChannelConfig cc;
    cc.channel = data->config.size()+1;
    cc.channelName = name;
    cc.enabled = enabled;
    cc.delay = delay;
    cc.width = width;
    cc.level = level;
    cc.role = role;
    if(role != BlackChirp::NoPulseRole)
        cc.channelName = BlackChirp::getPulseName(role);

    data->config.append(cc);
}

void PulseGenConfig::setRepRate(const double r)
{
    data->repRate = r;
}

