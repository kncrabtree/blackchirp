#include <data/experiment/flowconfig.h>

FlowConfig::FlowConfig() : data(new FlowConfigData)
{
}

FlowConfig::FlowConfig(const FlowConfig &rhs) : data(rhs.data)
{

}

FlowConfig &FlowConfig::operator=(const FlowConfig &rhs)
{
    if (this != &rhs)
        data.operator=(rhs.data);
    return *this;
}

FlowConfig::~FlowConfig()
{

}

QVariant FlowConfig::setting(int index, FlowChSetting s) const
{
    QVariant out;
    if(index < 0 || index > data->configList.size())
        return out;

    switch(s) {
    case Enabled:
        out = data->configList.at(index).enabled;
        break;
    case Setpoint:
        out = data->configList.at(index).setpoint;
        break;
    case Flow:
        out = data->configList.at(index).flow;
        break;
    case Name:
        out = data->configList.at(index).name;
        break;
    }

    return out;
}


double FlowConfig::pressureSetpoint() const
{
    return data->pressureSetpoint;
}

double FlowConfig::pressure() const
{
    return data->pressure;
}

bool FlowConfig::pressureControlMode() const
{
    return data->pressureControlMode;
}

int FlowConfig::size() const
{
    return data->configList.size();
}

void FlowConfig::add(double set, QString name)
{
    FlowChannel cc;
    cc.enabled = !qFuzzyCompare(1.0+set,1.0);
    cc.name = name;
    cc.setpoint = set;
    cc.flow = 0.0;
    data->configList.append(cc);
}

void FlowConfig::set(int index, FlowChSetting s, QVariant val)
{
    if(index < 0 || index > data->configList.size())
        return;

    switch(s) {
    case Enabled:
        //this is handled automatically by the setpoint case
        break;
    case Setpoint:
        data->configList[index].setpoint = val.toDouble();
        if(qFuzzyCompare(1.0+data->configList.at(index).setpoint,1.0))
            data->configList[index].enabled = false;
        else
            data->configList[index].enabled = true;
        break;
    case Flow:
        data->configList[index].flow = val.toDouble();
        break;
    case Name:
        data->configList[index].name = val.toString();
        break;
    }
}

void FlowConfig::setPressure(double p)
{
    data->pressure = p;
}


void FlowConfig::setPressureSetpoint(double s)
{
    data->pressureSetpoint = s;
}

void FlowConfig::setPressureControlMode(bool en)
{
    data->pressureControlMode = en;
}

QMap<QString, QPair<QVariant, QString> > FlowConfig::headerMap() const
{
    QMap<QString, QPair<QVariant, QString> > out;
    out.insert(QString("FlowConfigPressureControlMode"),qMakePair(pressureControlMode(),QString("")));
    out.insert(QString("FlowConfigPressureSetpoint"),qMakePair(QString::number(pressureSetpoint(),'f',3),QString("kTorr")));
    for(int i=0;i<data->configList.size(); i++)
    {
        if(data->configList.at(i).enabled)
        {
            out.insert(QString("FlowConfigChannel.%1.Name").arg(i),qMakePair(data->configList.at(i).name,QString("")));
            out.insert(QString("FlowConfigChannel.%1.Setpoint").arg(i),
                       qMakePair(QString::number(data->configList.at(i).setpoint,'f',2),QString("sccm")));
        }
    }

    return out;
}

void FlowConfig::parseLine(const QString key, const QVariant val)
{
    if(key.startsWith(QString("FlowConfig")))
    {
        if(key.endsWith(QString("PressureControlMode")))
            data->pressureControlMode = val.toBool();
        if(key.endsWith(QString("PressureSetpoint")))
            data->pressureSetpoint = val.toDouble();
        if(key.contains(QString("Channel")))
        {
            QStringList l = key.split(QString("."));
            if(l.size() < 3)
                return;

            QString subKey = l.constLast();
            int index = l.at(1).toInt();

            while(data->configList.size() <= index)
                data->configList.append(FlowChannel());

            if(subKey.contains(QString("Name")))
            {
                data->configList[index].enabled = true;
                data->configList[index].name = val.toString();
            }
            if(subKey.contains(QString("Setpoint")))
            {
                data->configList[index].enabled = true;
                data->configList[index].setpoint = val.toDouble();
            }
        }
    }
}

