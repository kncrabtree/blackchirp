#include <hardware/optional/flowcontroller/flowconfig.h>

#include <data/storage/settingsstorage.h>
#include <hardware/optional/flowcontroller/flowcontroller.h>

FlowConfig::FlowConfig(QString hwSubKey, int index) : HeaderStorage(BC::Key::hwKey(BC::Key::Flow::flowController,index),hwSubKey)
{
}

FlowConfig::~FlowConfig()
{

}

QVariant FlowConfig::setting(int index, FlowChSetting s) const
{
    QVariant out;
    if(index < 0 || index > d_configList.size())
        return out;

    switch(s) {
    case Enabled:
        out = d_configList.at(index).enabled;
        break;
    case Setpoint:
        out = d_configList.at(index).setpoint;
        break;
    case Flow:
        out = d_configList.at(index).flow;
        break;
    case Name:
        out = d_configList.at(index).name;
        break;
    }

    return out;
}

int FlowConfig::size() const
{
    return d_configList.size();
}

void FlowConfig::addCh(double set, QString name)
{
    FlowChannel cc;
    cc.enabled = !qFuzzyCompare(1.0+set,1.0);
    cc.name = name;
    cc.setpoint = set;
    cc.flow = 0.0;
    d_configList.append(cc);
}

void FlowConfig::setCh(int index, FlowChSetting s, QVariant val)
{
    if(index < 0 || index > d_configList.size())
        return;

    switch(s) {
    case Enabled:
        //this is handled automatically by the setpoint case
        break;
    case Setpoint:
        d_configList[index].setpoint = val.toDouble();
        if(qFuzzyCompare(1.0+d_configList.at(index).setpoint,1.0))
            d_configList[index].enabled = false;
        else
            d_configList[index].enabled = true;
        break;
    case Flow:
        d_configList[index].flow = val.toDouble();
        break;
    case Name:
        d_configList[index].name = val.toString();
        break;
    }
}

void FlowConfig::storeValues()
{
    //no need to store flow rates or actual pressure; those change with time and
    //don't make sense to record in the experiment header
    SettingsStorage s(BC::Key::Flow::flowController,SettingsStorage::Hardware);
    using namespace BC::Store::FlowConfig;

    store(pSetPoint,d_pressureSetpoint,s.get(BC::Key::Flow::pUnits,QString("")));
    store(pcEnabled,d_pressureControlMode);
    for(int i=0; i<d_configList.size(); ++i)
    {
        auto &fc = d_configList.at(i);
        storeArrayValue(channel,i,name,fc.name);
        storeArrayValue(channel,i,enabled,fc.enabled);
        storeArrayValue(channel,i,setPoint,fc.setpoint,s.getArrayValue(BC::Key::Flow::channels,i,BC::Key::Flow::chUnits,QString("")));
    }

}

void FlowConfig::retrieveValues()
{
     using namespace BC::Store::FlowConfig;

    d_pressureSetpoint = retrieve(pSetPoint,0.0);
    d_pressureControlMode = retrieve(pcEnabled,false);
    auto n = arrayStoreSize(channel);
    d_configList.clear();
    d_configList.reserve(n);
    for(std::size_t i=0; i<n; ++i)
    {
        FlowChannel cc {
            retrieveArrayValue(channel,i,enabled,false),
            retrieveArrayValue(channel,i,setPoint,0.0),
            0.0,
            retrieveArrayValue(channel,i,name,QString(""))
        };

        d_configList << cc;
    }
}
