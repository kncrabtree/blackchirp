#include <hardware/optional/pulsegenerator/pulsegenerator.h>

PulseGenerator::PulseGenerator(const QString subKey, const QString name, CommunicationProtocol::CommType commType, int numChannels, QObject *parent, bool threaded, bool critical) :
    HardwareObject(BC::Key::PGen::key,subKey,name,commType,parent,threaded,critical),
    d_numChannels(numChannels)
{
    SettingsStorage::set(BC::Key::PGen::numChannels,d_numChannels,true);
}

PulseGenerator::~PulseGenerator()
{

}

void PulseGenerator::initialize()
{
    for(int i=0; i<d_numChannels; i++)
        d_config.addChannel();

    initializePGen();
}

bool PulseGenerator::prepareForExperiment(Experiment &exp)
{
    if(exp.pGenConfig())
        return setAll(*exp.pGenConfig());

    return true;
}


void PulseGenerator::readChannel(const int index)
{
    auto c = d_config.settings(index);

    bool success = true;

    auto w = readChWidth(index);
    success &= (w <= get<double>(BC::Key::PGen::maxWidth) && w >= get<double>(BC::Key::PGen::minWidth));
    if(!success)
    {
        emit logMessage(QString("Could not read width for channel %1").arg(index),LogHandler::Error);
        return;
    }

    auto d = readChDelay(index);
    success &= (d <= get<double>(BC::Key::PGen::maxDelay) && d >= get<double>(BC::Key::PGen::minDelay));
    if(!success)
    {
        emit logMessage(QString("Could not read delay for channel %1").arg(index),LogHandler::Error);
        return;
    }

    auto en = readChEnabled(index);
    auto level = readChActiveLevel(index);

    if(success)
    {
        d_config.set(index,PulseGenConfig::WidthSetting,w);
        d_config.set(index,PulseGenConfig::DelaySetting,d);
        d_config.set(index,PulseGenConfig::EnabledSetting,en);
        d_config.set(index,PulseGenConfig::LevelSetting,level);
    }

}

double PulseGenerator::readRepRate()
{
    auto out = readHwRepRate();
    auto min = get(BC::Key::PGen::minRepRate,0.01);
    auto max = get(BC::Key::PGen::maxRepRate,1e5);
    if((out >= min) && (out <= max))
    {
        d_config.setRepRate(out);
        emit repRateUpdate(out,QPrivateSignal());
        return out;
    }

    if(!isnan(out))
        emit logMessage(QString("Rep rate (%1 Hz) is outside valid range (%2 - %3 Hz)").
                    arg(out,0,'e',2).arg(min,0,'e',2).arg(max,0,'e',2),LogHandler::Error);
    return nan("");
}

bool PulseGenerator::setPGenSetting(const int index, const PulseGenConfig::Setting s, const QVariant val)
{
    if(index >= d_config.size())
    {
        emit logMessage(QString("Received invalid channel (%1). Allowed values: 0-%2").arg(index).arg(d_config.size()-1),LogHandler::Error);
        return false;
    }

    bool success = true;
    QVariant result = val;
    switch(s) {
    case PulseGenConfig::NameSetting:
    case PulseGenConfig::RoleSetting:
        d_config.set(index,s,val);
        break;
    case PulseGenConfig::LevelSetting:
    {
        auto level = val.value<PulseGenConfig::ActiveLevel>();
        success = setChActiveLevel(index,level);
        auto a = readChActiveLevel(index);
        if(a == level)
            result = a;
        else
            success = false;

        break;
    }
    case PulseGenConfig::EnabledSetting:
    {
        auto en = val.toBool();
        success = setChEnabled(index,en);
        en = readChEnabled(index);
        if(en == val.toBool())
            result = en;
        else
            success = false;
        break;
    }
    case PulseGenConfig::WidthSetting:
    {
        auto w = val.toDouble();
        auto min = get<double>(BC::Key::PGen::minWidth);
        auto max = get<double>(BC::Key::PGen::maxWidth);
        if((w < min) || (w > max))
        {
            success = false;
            emit logMessage(QString("Requested width (%1) for channel %2 is outside the allowed range (%3 - %4)").arg(w,0,'e',2).arg(index).arg(min,0,'e',2).arg(max,0,'e',2),LogHandler::Error);
        }
        else
        {
            success = setChWidth(index,w);
            if(success)
            {
                double newW = readChWidth(index);
                if(fabs(newW - w) <= min)
                    result = newW;
                else
                    success = false;

            }
        }
        break;
    }
    case PulseGenConfig::DelaySetting:
    {
        auto d = val.toDouble();
        auto min = get<double>(BC::Key::PGen::minDelay);
        auto max = get<double>(BC::Key::PGen::maxDelay);
        if((d < min) || (d > max))
        {
            success = false;
            emit logMessage(QString("Requested delay (%1) for channel %2 is outside the allowed range (%3 - %4)").arg(d,0,'e',2).arg(index).arg(min,0,'e',2).arg(max,0,'e',2),LogHandler::Error);
        }
        else
        {
            success = setChDelay(index,d);
            if(success)
            {
                double newD = readChDelay(index);
                if(fabs(newD - d) <= min)
                    result = newD;
                else
                    success = false;

            }
        }
        break;
    }
    default:
        break;
    }

    if(success)
    {
        d_config.set(index,s,result);
        emit settingUpdate(index,s,result,QPrivateSignal());
    }

    return success;
}


bool PulseGenerator::setChannel(const int index, const PulseGenConfig::ChannelConfig &cc)
{
    bool success = true;

    blockSignals(true);
    success &= setPGenSetting(index,PulseGenConfig::NameSetting,cc.channelName);
    if(success)
        success &= setPGenSetting(index,PulseGenConfig::EnabledSetting,cc.enabled);
    if(success)
        success &= setPGenSetting(index,PulseGenConfig::DelaySetting,cc.delay);
    if(success)
        success &= setPGenSetting(index,PulseGenConfig::WidthSetting,cc.width);
    if(success)
        success &= setPGenSetting(index,PulseGenConfig::LevelSetting,cc.level);
    if(success)
        setPGenSetting(index,PulseGenConfig::RoleSetting,cc.role);

    blockSignals(false);
    if(success)
        emit configUpdate(d_config,QPrivateSignal());

    return success;
}

bool PulseGenerator::setAll(const PulseGenConfig &cc)
{
    blockSignals(true);
    bool success = true;
    for(int i=0; i<d_config.size(); i++)
    {
        success &= setChannel(i,cc.at(i));
        if(!success)
            break;
    }

    if(success)
        success &= setRepRate(cc.repRate());
    blockSignals(false);

    if(success)
        emit configUpdate(d_config,QPrivateSignal());

    return success;
}

bool PulseGenerator::setRepRate(double d)
{
    auto min = get(BC::Key::PGen::minRepRate,0.01);
    auto max = get(BC::Key::PGen::maxRepRate,1e5);
    if((d < min) || (d > max))
    {
        emit logMessage(QString("Requested rep rate (%1 Hz) is outside the allowed range (%2 - %3 Hz)").
                        arg(d,0,'e',2).arg(min,0,'e',2).arg(max,0,'e',2),LogHandler::Error);
        return false;
    }
    bool success = setHwRepRate(d);
    double rr = d;
    if(success)
        rr = readRepRate();

    return (success && (!isnan(rr)));

}

#ifdef BC_LIF
bool PulseGenerator::setLifDelay(double d)
{
    bool success = false;
    auto l = d_config.channelsForRole(PulseGenConfig::LIF);
    for(int i=0; i<l.size(); i++)
    {
        if(!setPGenSetting(l.at(i),PulseGenConfig::DelaySetting,d))
            return false;
        else
            success = true;
    }

    return success;
}
#endif

void PulseGenerator::readAll()
{
    blockSignals(true);
    for(int i=0;i<d_numChannels; i++)
        readChannel(i);

    readRepRate();
    blockSignals(false);

    emit configUpdate(d_config,QPrivateSignal());
}


QStringList PulseGenerator::forbiddenKeys() const
{
    return {BC::Key::PGen::numChannels};
}
