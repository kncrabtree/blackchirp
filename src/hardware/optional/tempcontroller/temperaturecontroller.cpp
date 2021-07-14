#include <hardware/optional/tempcontroller/temperaturecontroller.h>

#include <QTimer>

using namespace BC::Key::TC;

TemperatureController::TemperatureController(const QString subKey, const QString name, CommunicationProtocol::CommType commType, int numChannels, QObject *parent, bool threaded, bool critical) :
    HardwareObject(key,subKey,name,commType,parent,threaded,critical), d_numChannels(numChannels)
{
    d_temperatureList.clear();
    for (int i=0; i<d_numChannels;i++)
        d_temperatureList.append(0.0);

    if(!containsArray(channels))
    {
        std::vector<SettingsMap> l;
        l.reserve(d_numChannels);
        for(int i=0; i<d_numChannels; ++i)
            l.push_back({
                            {name,QString("Ch")+QString::number(i+1)},
                            {enabled,true},
                            {decimals,4}
                        });
        setArray(channels,l);
    }
}

TemperatureController::~TemperatureController()
{
}

QList<double> TemperatureController::readAll()
{
   d_temperatureList = readHWTemperatures();
   if(d_temperatureList.size() == d_numChannels)
       emit temperatureListUpdate(d_temperatureList, QPrivateSignal());
   return d_temperatureList;
}

double TemperatureController::readTemperature(const int ch)
{
    auto t = readHwTemperature(ch);
    if(!isnan(t) && ch >=0 && ch < d_temperatureList.size())
    {
        d_temperatureList[ch] = t;
        emit temperatureUpdate(ch,t,QPrivateSignal());
    }

    return t;
}

bool TemperatureController::prepareForExperiment(Experiment &e)
{
    for (int i=0;i<d_temperatureList.size();i++)
        e.auxData()->registerKey(d_key,d_subKey,BC::Aux::TC::temperature.arg(i));

    return true;
}

AuxDataStorage::AuxDataMap TemperatureController::readAuxData()
{
    AuxDataStorage::AuxDataMap out;
    for (int i=0;i<d_temperatureList.size();i++)
        out.insert({BC::Aux::TC::temperature.arg(i),d_temperatureList.at(i)});
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
