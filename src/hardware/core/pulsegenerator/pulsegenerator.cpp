#include <src/hardware/core/pulsegenerator/pulsegenerator.h>

PulseGenerator::PulseGenerator(QObject *parent) :
   HardwareObject(parent), d_minWidth(0.010), d_maxWidth(100000.0), d_minDelay(0.0), d_maxDelay(100000.0)
{
    d_key = QString("pGen");
}

PulseGenerator::~PulseGenerator()
{

}

void PulseGenerator::initialize()
{
    //set up config
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(d_key);
    s.beginGroup(d_subKey);

    s.beginReadArray(QString("channels"));
    for(int i=0; i<d_numChannels; i++)
    {
        s.setArrayIndex(i);
        QString name = s.value(QString("name"),QString("Ch%1").arg(i)).toString();
        double d = s.value(QString("defaultDelay"),0.0).toDouble();
        double w = s.value(QString("defaultWidth"),0.050).toDouble();
        QVariant lvl = s.value(QString("level"),BlackChirp::PulseLevelActiveHigh);
        bool en = s.value(QString("defaultEnabled"),false).toBool();
        auto role = static_cast<BlackChirp::PulseRole>(s.value(QString("role"),BlackChirp::NoPulseRole).toInt());

        if(lvl == QVariant(BlackChirp::PulseLevelActiveHigh))
            d_config.add(name,en,d,w,BlackChirp::PulseLevelActiveHigh,role);
        else
            d_config.add(name,en,d,w,BlackChirp::PulseLevelActiveLow,role);
    }
    s.endArray();

    d_config.setRepRate(s.value(QString("repRate"),10.0).toDouble());
    s.endGroup();
    s.endGroup();

    initializePGen();
}

Experiment PulseGenerator::prepareForExperiment(Experiment exp)
{
    bool success = setAll(exp.pGenConfig());
    if(!success)
        exp.setHardwareFailed();

    return exp;
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


BlackChirp::PulseChannelConfig PulseGenerator::read(const int index)
{
    BlackChirp::PulseChannelConfig out = d_config.settings(index);
    bool ok = false;
    out.channelName = read(index,BlackChirp::PulseNameSetting).toString();
    out.delay = read(index,BlackChirp::PulseDelaySetting).toDouble(&ok);
    if(!ok)
        return out;
    out.width = read(index,BlackChirp::PulseWidthSetting).toDouble(&ok);
    if(!ok)
        return out;
    out.enabled = read(index,BlackChirp::PulseEnabledSetting).toBool();
    out.level = read(index,BlackChirp::PulseLevelSetting).value<BlackChirp::PulseActiveLevel>();
    out.role = static_cast<BlackChirp::PulseRole>(read(index,BlackChirp::PulseRoleSetting).toInt());

    return out;
}


bool PulseGenerator::setChannel(const int index, const BlackChirp::PulseChannelConfig cc)
{
    bool success = true;

    success &= set(index,BlackChirp::PulseNameSetting,cc.channelName);
    success &= set(index,BlackChirp::PulseEnabledSetting,cc.enabled);
    success &= set(index,BlackChirp::PulseDelaySetting,cc.delay);
    success &= set(index,BlackChirp::PulseWidthSetting,cc.width);
    success &= set(index,BlackChirp::PulseLevelSetting,cc.level);
    set(index,BlackChirp::PulseRoleSetting,cc.role);

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
    auto l = d_config.channelsForRole(BlackChirp::LifPulseRole);
    for(int i=0; i<l.size(); i++)
    {
        if(!set(l.at(i),BlackChirp::PulseDelaySetting,d))
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
