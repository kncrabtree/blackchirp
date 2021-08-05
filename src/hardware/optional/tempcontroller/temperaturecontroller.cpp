#include <hardware/optional/tempcontroller/temperaturecontroller.h>

#include <QTimer>

using namespace BC::Key::TC;

TemperatureController::TemperatureController(const QString subKey, const QString name, CommunicationProtocol::CommType commType, int numChannels, QObject *parent, bool threaded, bool critical) :
    HardwareObject(key,subKey,name,commType,parent,threaded,critical), d_numChannels(numChannels)
{
    d_config.setNumChannels(d_numChannels);

    if(!containsArray(channels))
    {
        std::vector<SettingsMap> l;
        l.reserve(d_numChannels);
        for(int i=0; i<d_numChannels; ++i)
            l.push_back({
                            {chName,QString("Temperature Ch")+QString::number(i+1)},
                            {enabled,false},
                            {decimals,4}
                        });
        setArray(channels,l);
    }

    for(int i=0; i<d_numChannels; ++i)
    {
        d_config.setEnabled(i,getArrayValue(channels,i,enabled,false));
        d_config.setName(i,getArrayValue(channels,i,chName,QString("")));
    }
}

TemperatureController::~TemperatureController()
{
}

void TemperatureController::readAll()
{
    for(int i=0; i<d_numChannels; ++i)
    {
        if(d_config.channelEnabled(i))
        {
            if(isnan(readTemperature(i)))
                break;
        }
    }
}

double TemperatureController::readTemperature(const int ch)
{
    auto t = readHwTemperature(ch);
    if(!isnan(t))
    {
        d_config.setTemperature(ch,t);
        emit temperatureUpdate(ch,t,QPrivateSignal());
    }

    return t;
}

bool TemperatureController::prepareForExperiment(Experiment &e)
{

    for (int i=0;i<d_numChannels;i++)
    {
        if(e.tcConfig())
        {
            d_config.setName(i,e.tcConfig()->channelName(i));
            d_config.setEnabled(i,e.tcConfig()->channelEnabled(i));
        }

        e.auxData()->registerKey(d_key,d_subKey,BC::Aux::TC::temperature.arg(i));
    }

    e.setTempControllerConfig(d_config);
    return true;
}

AuxDataStorage::AuxDataMap TemperatureController::readAuxData()
{
    AuxDataStorage::AuxDataMap out;
    for (int i=0;i<d_numChannels;i++)
    {
        if(d_config.channelEnabled(i))
        {
            auto n = d_config.channelName(i);
            if(n.isEmpty())
                out.insert({BC::Aux::TC::temperature.arg(i),d_config.temperature(i)});
            else
                out.insert({n+"."+BC::Aux::TC::temperature.arg(i),d_config.temperature(i)});
        }
    }
    return out;
}

void TemperatureController::initialize()
{
    p_readTimer = new QTimer(this);
    p_readTimer->setInterval(get(interval,500));
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

void TemperatureController::poll()
{
    readAll();
}


QStringList TemperatureController::validationKeys() const
{
    QStringList out;
    for(int i=0; i<d_numChannels; ++i)
        out.append(BC::Aux::TC::temperature.arg(i));

    return out;
}
