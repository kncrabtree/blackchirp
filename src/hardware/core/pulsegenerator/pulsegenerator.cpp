#include <src/hardware/core/pulsegenerator/pulsegenerator.h>

PulseGenerator::PulseGenerator(const QString subKey, const QString name, CommunicationProtocol::CommType commType, int numChannels, QObject *parent, bool threaded, bool critical) :
    HardwareObject(BC::Key::PGen::key,subKey,name,commType,parent,threaded,critical),
    d_minWidth(0.010), d_maxWidth(100000.0), d_minDelay(0.0), d_maxDelay(100000.0),
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
    bool success = setAll(exp.pGenConfig());
    if(!success)
        exp.setHardwareFailed();

    return exp.hardwareSuccess();
}

void PulseGenerator::readSettings()
{
    QSettings s(QSettings::SystemScope, QApplication::organizationName(), QApplication::applicationName());
    s.beginGroup(d_key);
    s.beginGroup(d_subKey);

    d_minWidth = s.value(QString("minWidth"),0.004).toDouble();
    d_maxWidth = s.value(QString("maxWidth"),100000.0).toDouble();
    d_minDelay = s.value(QString("minDelay"),0.0).toDouble();
    d_maxDelay = s.value(QString("maxDelay"),100000.0).toDouble();

    s.setValue(QString("minWidth"),d_minWidth);
    s.setValue(QString("maxWidth"),d_maxWidth);
    s.setValue(QString("minDelay"),d_minDelay);
    s.setValue(QString("maxDelay"),d_maxDelay);

    s.endGroup();
    s.endGroup();
    s.sync();
}


PulseGenConfig::ChannelConfig PulseGenerator::read(const int index)
{
    PulseGenConfig::ChannelConfig out = d_config.settings(index);
    bool ok = false;
    out.delay = read(index,PulseGenConfig::DelaySetting).toDouble(&ok);
    if(!ok)
        return out;
    out.width = read(index,PulseGenConfig::WidthSetting).toDouble(&ok);
    if(!ok)
        return out;
    out.enabled = read(index,PulseGenConfig::EnabledSetting).toBool();
    out.level = read(index,PulseGenConfig::LevelSetting).value<PulseGenConfig::ActiveLevel>();

    return out;
}


bool PulseGenerator::setChannel(const int index, const PulseGenConfig::ChannelConfig &cc)
{
    bool success = true;

    success &= set(index,PulseGenConfig::NameSetting,cc.channelName);
    success &= set(index,PulseGenConfig::EnabledSetting,cc.enabled);
    success &= set(index,PulseGenConfig::DelaySetting,cc.delay);
    success &= set(index,PulseGenConfig::WidthSetting,cc.width);
    success &= set(index,PulseGenConfig::LevelSetting,cc.level);
    set(index,PulseGenConfig::RoleSetting,cc.role);

    return success;
}

bool PulseGenerator::setAll(const PulseGenConfig cc)
{
    bool success = true;
    for(int i=0; i<d_config.size(); i++)
        success &= setChannel(i,cc.at(i));

    success &= setRepRate(cc.repRate());

    return success;
}

#ifdef BC_LIF
bool PulseGenerator::setLifDelay(double d)
{
    bool success = false;
    auto l = d_config.channelsForRole(PulseGenConfig::LifRole);
    for(int i=0; i<l.size(); i++)
    {
        if(!set(l.at(i),PulseGenConfig::DelaySetting,d))
            return false;
        else
            success = true;
    }

    return success;
}
#endif

void PulseGenerator::readAll()
{
    for(int i=0;i<d_numChannels; i++)
        read(i);

    readRepRate();
}
