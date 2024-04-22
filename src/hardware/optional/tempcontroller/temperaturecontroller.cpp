#include <hardware/optional/tempcontroller/temperaturecontroller.h>

#include <QTimer>

using namespace BC::Key::TC;

TemperatureController::TemperatureController(const QString subKey, const QString name, CommunicationProtocol::CommType commType, uint numChannels, QObject *parent, bool threaded, bool critical) :
    HardwareObject(key,subKey,name,commType,parent,threaded,critical,d_count), d_numChannels(numChannels), d_config{subKey,d_count}
{
    d_config.setNumChannels(d_numChannels);

    set(::numChannels,d_numChannels);
    setDefault(interval,500);

    if(!containsArray(channels))
    {
        std::vector<SettingsMap> l;
        l.reserve(d_numChannels);
        for(uint i=0; i<d_numChannels; ++i)
            l.push_back({
                            {chName,QString("")},
                            {enabled,false},
                            {decimals,4},
                            {units,QString("K")}
                        });
        setArray(channels,l,true);
    }

    for(uint i=0; i<d_numChannels; ++i)
    {
        d_config.setEnabled(i,getArrayValue(channels,i,enabled,false));
        setChannelName(i,getArrayValue(channels,i,chName,QString("")));
    }

    d_count++;
}

TemperatureController::~TemperatureController()
{
    for(uint i=0; i<d_numChannels; ++i)
    {
        setArrayValue(channels,i,chName,d_config.channelName(i));
        setArrayValue(channels,i,enabled,d_config.channelEnabled(i));
    }
}

void TemperatureController::readAll()
{
    for(uint i=0; i<d_numChannels; ++i)
    {
        if(readChannelEnabled(i))
        {
            if(isnan(readTemperature(i)))
                break;
        }
    }
}

void TemperatureController::setChannelEnabled(uint ch, bool en)
{
    setHwChannelEnabled(ch,en);
    readChannelEnabled(ch);
}

void TemperatureController::setChannelName(uint ch, const QString name)
{
    if(!name.isEmpty())
        d_config.setName(ch,name);
    else
        d_config.setName(ch,QString("Temperature Ch%1").arg(ch+1));
}

double TemperatureController::readTemperature(const uint ch)
{
    auto t = readHwTemperature(ch);
    if(!isnan(t))
    {
        d_config.setTemperature(ch,t);
        emit temperatureUpdate(ch,t,QPrivateSignal());
    }

    return t;
}

bool TemperatureController::readChannelEnabled(const uint ch)
{
    auto out = readHwChannelEnabled(ch);
    emit channelEnableUpdate(ch,out,QPrivateSignal());
    return out;
}

bool TemperatureController::prepareForExperiment(Experiment &e)
{
    auto wp = e.getOptHwConfig<TemperatureControllerConfig>(d_config.headerKey());

    for (uint i=0;i<d_numChannels;i++)
    {
        if(auto p = wp.lock())
        {
            d_config.setName(i,p->channelName(i));
            setChannelEnabled(i,p->channelEnabled(i));
        }

        e.auxData()->registerKey(d_key,d_subKey,BC::Aux::TC::temperature.arg(i));
    }

    e.addOptHwConfig(d_config);

    return true;
}

AuxDataStorage::AuxDataMap TemperatureController::readAuxData()
{
    AuxDataStorage::AuxDataMap out;
    for (uint i=0;i<d_numChannels;i++)
    {
        if(d_config.channelEnabled(i))
        {
            auto n = d_config.channelName(i);
            if(n.isEmpty())
                out.insert({BC::Aux::TC::temperature.arg(i+1),d_config.temperature(i)});
            else
                out.insert({n+"."+BC::Aux::TC::temperature.arg(i+1),d_config.temperature(i)});
        }
    }
    return out;
}

void TemperatureController::initialize()
{
    p_readTimer = new QTimer(this);
    connect(p_readTimer,&QTimer::timeout,this,&TemperatureController::poll);
    connect(this,&TemperatureController::hardwareFailure,p_readTimer,&QTimer::stop);
    tcInitialize();
}

bool TemperatureController::testConnection()
{
    p_readTimer->stop();
    if(!tcTestConnection())
        return false;

    readAll();
    p_readTimer->start();
    return true;
}

void TemperatureController::readSettings()
{
    p_readTimer->setInterval(get(interval,500));
}

void TemperatureController::poll()
{
    readAll();
}


QStringList TemperatureController::validationKeys() const
{
    QStringList out;
    for(uint i=0; i<d_numChannels; ++i)
        out.append(BC::Aux::TC::temperature.arg(i));

    return out;
}


QStringList TemperatureController::forbiddenKeys() const
{
    using namespace BC::Key::TC;
    return {chName,enabled,::numChannels};
}
