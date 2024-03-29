#include <hardware/optional/pulsegenerator/pulsegenerator.h>

#include <gui/widget/pulseconfigwidget.h>

PulseGenerator::PulseGenerator(const QString subKey, const QString name, CommunicationProtocol::CommType commType, int numChannels, QObject *parent, bool threaded, bool critical) :
    HardwareObject(BC::Key::PGen::key,subKey,name,commType,parent,threaded,critical),
    d_numChannels(numChannels)
{
    SettingsStorage::set(BC::Key::PGen::numChannels,d_numChannels,true);

    for(int i=0; i<d_numChannels; i++)
        addChannel();

    using namespace BC::Key::PulseWidget;
    SettingsStorage s(key);
    if(s.containsArray(channels))
    {
        for(int i=0; i<d_numChannels; i++)
        {
            setCh(i,PulseGenConfig::NameSetting,s.getArrayValue(channels,i,BC::Key::PulseWidget::name,QString("Ch%1").arg(i+1)));
            setCh(i,PulseGenConfig::RoleSetting,s.getArrayValue(channels,i,role,PulseGenConfig::None));
        }
    }
}

PulseGenerator::~PulseGenerator()
{

}

void PulseGenerator::initialize()
{
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
    auto c = settings(index);

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
    auto mode = readChMode(index);
    auto sync = readChSynchCh(index);
    auto dutyOn = readChDutyOn(index);
    auto dutyOff = readChDutyOff(index);

    if(success)
    {
        setCh(index,PulseGenConfig::WidthSetting,w);
        setCh(index,PulseGenConfig::DelaySetting,d);
        setCh(index,PulseGenConfig::EnabledSetting,en);
        setCh(index,PulseGenConfig::LevelSetting,level);
        setCh(index,PulseGenConfig::ModeSetting,mode);
        setCh(index,PulseGenConfig::SyncSetting,sync);
        setCh(index,PulseGenConfig::DutyOnSetting,dutyOn);
        setCh(index,PulseGenConfig::DutyOffSetting,dutyOff);
    }

}

double PulseGenerator::readRepRate()
{
    auto out = readHwRepRate();
    auto min = get(BC::Key::PGen::minRepRate,0.01);
    auto max = get(BC::Key::PGen::maxRepRate,1e5);
    if((out >= min) && (out <= max))
    {
        d_repRate = out;
        emit repRateUpdate(out,QPrivateSignal());
        return out;
    }

    if(!isnan(out))
        emit logMessage(QString("Rep rate (%1 Hz) is outside valid range (%2 - %3 Hz)").
                    arg(out,0,'e',2).arg(min,0,'e',2).arg(max,0,'e',2),LogHandler::Error);
    return nan("");
}

PulseGenConfig::PGenMode PulseGenerator::readPulseMode()
{
    auto out = readHwPulseMode();
    d_mode = out;
    emit modeUpdate(out,QPrivateSignal());
    return out;
}

bool PulseGenerator::readPulseEnabled()
{
    auto out = readHwPulseEnabled();
    d_pulseEnabled = out;
    emit pulseEnabledUpdate(out,QPrivateSignal());
    return out;
}

bool PulseGenerator::setPGenSetting(const int index, const PulseGenConfig::Setting s, const QVariant val)
{
    if(index >= d_channels.size())
    {
        emit logMessage(QString("Received invalid channel (%1). Allowed values: 0-%2").arg(index).arg(d_channels.size()-1),LogHandler::Error);
        return false;
    }

    bool success = true;
    QVariant result = val;
    switch(s) {
    case PulseGenConfig::NameSetting:
    case PulseGenConfig::RoleSetting:
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
    case PulseGenConfig::ModeSetting:
    {
        auto d = val.value<PulseGenConfig::ChannelMode>();
        auto ok = get(BC::Key::PGen::canDutyCycle,false);
        if(!ok && d == PulseGenConfig::DutyCycle)
        {
            success = false;
            emit logMessage("Duty cycle mode is not supported.",LogHandler::Error);
        }
        else
        {
            success = setChMode(index,d);
            if(success)
            {
                auto m = readChMode(index);
                if(m == d)
                    result = m;
                else
                    success = false;
            }
        }
        break;
    }
    case PulseGenConfig::SyncSetting:
    {
        auto d = val.toInt();
        auto ok = get(BC::Key::PGen::canSyncToChannel,false);
        if(!ok && d != 0)
        {
            success = false;
            emit logMessage(QString("Syncing one channel to another is not supported."),LogHandler::Error);
        }
        else if (d < 0 || d > d_numChannels)
        {
            success = false;
            emit logMessage(QString("Requested sync channel (%1) is invalid").arg(d),LogHandler::Error);
        }
        else
        {
            success = setChSyncCh(index,d);
            if(success)
            {
                auto ch = readChSynchCh(index);
                if(d == ch)
                    result = ch;
                else
                    success = false;
            }
        }
        break;
    }
    case PulseGenConfig::DutyOnSetting:
    {
        auto d = val.toInt();
        auto max = get(BC::Key::PGen::dutyMax,1000);
        if(d<1 || d > max)
        {
            success = false;
            emit logMessage(QString("Requested number of duty cycle on pulses (%1) exceeds the maximum limit of %2.").arg(d).arg(max),LogHandler::Error);
        }
        else
        {
            success = setChDutyOn(index,d);
            if(success)
            {
                auto dd = readChDutyOn(index);
                if(d == dd)
                    result = dd;
                else
                    success = false;
            }
        }
        break;
    }
    case PulseGenConfig::DutyOffSetting:
    {
        auto d = val.toInt();
        auto max = get(BC::Key::PGen::dutyMax,1000);
        if(d<1 || d > max)
        {
            success = false;
            emit logMessage(QString("Requested number of duty cycle off pulses (%1) exceeds the maximum limit of %2.").arg(d).arg(max),LogHandler::Error);
        }
        else
        {
            success = setChDutyOff(index,d);
            if(success)
            {
                auto dd = readChDutyOff(index);
                if(d == dd)
                    result = dd;
                else
                    success = false;
            }
        }
        break;
    }
    }

    if(success)
    {
        setCh(index,s,result);
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
    if(success)
        success &= setPGenSetting(index,PulseGenConfig::ModeSetting,cc.mode);
    if(success)
        success &= setPGenSetting(index,PulseGenConfig::SyncSetting,cc.syncCh);
    if(success)
        success &= setPGenSetting(index,PulseGenConfig::DutyOnSetting,cc.dutyOn);
    if(success)
        success &= setPGenSetting(index,PulseGenConfig::DutyOffSetting,cc.dutyOff);

    blockSignals(false);
    if(success)
        emit configUpdate(static_cast<PulseGenConfig>(*this),QPrivateSignal());

    return success;
}

bool PulseGenerator::setAll(const PulseGenConfig &cc)
{
    blockSignals(true);
    bool success = true;
    for(int i=0; i<d_channels.size(); i++)
    {
        success &= setChannel(i,cc.at(i));
        if(!success)
            break;
    }

    if(success)
        success &= setRepRate(cc.d_repRate);
//    if(success)
//        success &= setPulseMode(cc.d_mode);
    if(success)
        success &= setPulseEnabled(cc.d_pulseEnabled);
    blockSignals(false);

    if(success)
        emit configUpdate(static_cast<PulseGenConfig>(*this),QPrivateSignal());

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

bool PulseGenerator::setPulseMode(PGenMode mode)
{
    auto ok = get(BC::Key::PGen::canTrigger,false);
    if(!ok && mode == PulseGenConfig::Triggered)
    {
        emit logMessage("Triggered mode is not supported.",LogHandler::Error);
        return false;
    }

    bool success = setHwPulseMode(mode);
    auto m = mode;
    if(success)
        m = readPulseMode();

    return (success && (m == mode));
}

bool PulseGenerator::setPulseEnabled(bool en)
{
    bool success = setHwPulseEnabled(en);
    bool e = false;
    if(success)
        e = readPulseEnabled();

    return (success && (e == en));
}

#ifdef BC_LIF
bool PulseGenerator::setLifDelay(double d)
{
    auto i = channelForRole(PulseGenConfig::LIF);
    if(i>=0)
        return setPGenSetting(i,PulseGenConfig::DelaySetting,d);
    else
        emit logMessage("Cannot set LIF delay; no channel configured for LIF role.", LogHandler::Error);

    return false;
}
#endif

void PulseGenerator::readAll()
{
    blockSignals(true);
    for(int i=0;i<d_numChannels; i++)
        readChannel(i);

    readRepRate();
    readPulseMode();
    readPulseEnabled();
    blockSignals(false);

    emit configUpdate(static_cast<PulseGenConfig&>(*this),QPrivateSignal());
}


QStringList PulseGenerator::forbiddenKeys() const
{
    return {BC::Key::PGen::numChannels};
}
