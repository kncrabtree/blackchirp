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

QVariant PulseGenConfig::setting(const int index, const PulseGenConfig::Setting s) const
{
    if(index < 0 || index >= data->config.size())
        return QVariant();

    switch(s)
    {
    case Delay:
        return data->config.at(index).delay;
        break;
    case Width:
        return data->config.at(index).width;
        break;
    case Enabled:
        return data->config.at(index).enabled;
        break;
    case Level:
        return data->config.at(index).level;
        break;
    case Name:
        return data->config.at(index).channelName;
        break;
    default:
        break;
    }

    return QVariant();
}

PulseGenConfig::ChannelConfig PulseGenConfig::settings(const int index) const
{
    if(index < 0 || index >= data->config.size())
        return ChannelConfig();

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
        out.insert(QString("PulseGen.%1.Delay").arg(i),qMakePair(QString::number(data->config.at(i).delay,'f',3),QString("")));
        out.insert(QString("PulseGen.%1.Width").arg(i),qMakePair(QString::number(data->config.at(i).width,'f',3),QString("")));
        out.insert(QString("PulseGen.%1.Level").arg(i),qMakePair(data->config.at(i).level,QString("")));
    }

    return out;
}

void PulseGenConfig::set(const int index, const PulseGenConfig::Setting s, const QVariant val)
{
    if(index < 0 || index >= data->config.size())
        return;

    switch(s)
    {
    case Delay:
        data->config[index].delay = val.toDouble();
        break;
    case Width:
        data->config[index].width = val.toDouble();
        break;
    case Enabled:
        data->config[index].enabled = val.toBool();
        break;
    case Level:
        data->config[index].level = val.value<PulseGenConfig::ActiveLevel>();
        break;
    case Name:
        data->config[index].channelName = val.toString();
        break;
    default:
        break;
    }
}

void PulseGenConfig::set(const int index, const PulseGenConfig::ChannelConfig cc)
{
    if(index < 0 || index >= data->config.size())
        return;

    set(index,Delay,cc.delay);
    set(index,Width,cc.width);
    set(index,Enabled,cc.enabled);
    set(index,Level,cc.level);
    set(index,Name,cc.channelName);
}

void PulseGenConfig::add(const QString name, const bool enabled, const double delay, const double width, const PulseGenConfig::ActiveLevel level)
{
    ChannelConfig cc;
    cc.channel = data->config.size()+1;
    cc.channelName = name;
    cc.enabled = enabled;
    cc.delay = delay;
    cc.width = width;
    cc.level = level;

    data->config.append(cc);
}

void PulseGenConfig::setRepRate(const double r)
{
    data->repRate = r;
}

