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
    case BlackChirp::PulseDelay:
        return data->config.at(index).delay;
        break;
    case BlackChirp::PulseWidth:
        return data->config.at(index).width;
        break;
    case BlackChirp::PulseEnabled:
        return data->config.at(index).enabled;
        break;
    case BlackChirp::PulseLevel:
        return data->config.at(index).level;
        break;
    case BlackChirp::PulseName:
        return data->config.at(index).channelName;
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
                data->config[index].channelName = val.toString();
            if(subKey.endsWith(QString("Enabled")))
                data->config[index].enabled = val.toBool();
            if(subKey.endsWith(QString("Delay")))
                data->config[index].delay = val.toDouble();
            if(subKey.endsWith(QString("Width")))
                data->config[index].width = val.toDouble();
            if(subKey.endsWith(QString("Level")))
                data->config[index].level = (BlackChirp::PulseActiveLevel)val.toInt();
        }
    }
}

void PulseGenConfig::set(const int index, const BlackChirp::PulseSetting s, const QVariant val)
{
    if(index < 0 || index >= data->config.size())
        return;

    switch(s)
    {
    case BlackChirp::PulseDelay:
        data->config[index].delay = val.toDouble();
        break;
    case BlackChirp::PulseWidth:
        data->config[index].width = val.toDouble();
        break;
    case BlackChirp::PulseEnabled:
        data->config[index].enabled = val.toBool();
        break;
    case BlackChirp::PulseLevel:
        data->config[index].level = static_cast<BlackChirp::PulseActiveLevel>(val.toInt());
        break;
    case BlackChirp::PulseName:
        data->config[index].channelName = val.toString();
        break;
    default:
        break;
    }
}

void PulseGenConfig::set(const int index, const BlackChirp::PulseChannelConfig cc)
{
    if(index < 0 || index >= data->config.size())
        return;

    set(index,BlackChirp::PulseDelay,cc.delay);
    set(index,BlackChirp::PulseWidth,cc.width);
    set(index,BlackChirp::PulseEnabled,cc.enabled);
    set(index,BlackChirp::PulseLevel,cc.level);
    set(index,BlackChirp::PulseName,cc.channelName);
}

void PulseGenConfig::add(const QString name, const bool enabled, const double delay, const double width, const BlackChirp::PulseActiveLevel level)
{
    BlackChirp::PulseChannelConfig cc;
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

