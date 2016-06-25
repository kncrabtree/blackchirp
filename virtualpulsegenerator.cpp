#include "virtualpulsegenerator.h"

VirtualPulseGenerator::VirtualPulseGenerator(QObject *parent) : PulseGenerator(parent)
{
    d_subKey = QString("virtual");
    d_prettyName = QString("Virtual Pulse Generator");
    d_commType = CommunicationProtocol::Virtual;
    d_threaded = false;

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

VirtualPulseGenerator::~VirtualPulseGenerator()
{

}



bool VirtualPulseGenerator::testConnection()
{
    blockSignals(true);
    readAll();
    blockSignals(false);

    emit configUpdate(d_config);
    emit connected();
    return true;
}

void VirtualPulseGenerator::initialize()
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(d_key);
    s.beginGroup(d_subKey);

    s.beginReadArray(QString("channels"));
    for(int i=0; i<BC_PGEN_NUMCHANNELS; i++)
    {
        s.setArrayIndex(i);
        QString name = s.value(QString("name"),QString("Ch%1").arg(i)).toString();
        double d = s.value(QString("defaultDelay"),0.0).toDouble();
        double w = s.value(QString("defaultWidth"),0.050).toDouble();
        QVariant lvl = s.value(QString("level"),BlackChirp::PulseLevelActiveHigh);
        bool en = s.value(QString("defaultEnabled"),false).toBool();

        if(lvl == QVariant(BlackChirp::PulseLevelActiveHigh))
            d_config.add(name,en,d,w,BlackChirp::PulseLevelActiveHigh);
        else
            d_config.add(name,en,d,w,BlackChirp::PulseLevelActiveLow);
    }
    s.endArray();

    d_config.setRepRate(s.value(QString("repRate"),10.0).toDouble());
    s.endGroup();
    s.endGroup();

    testConnection();
}

Experiment VirtualPulseGenerator::prepareForExperiment(Experiment exp)
{
    setAll(exp.pGenConfig());
    return exp;
}

void VirtualPulseGenerator::beginAcquisition()
{
}

void VirtualPulseGenerator::endAcquisition()
{
}

void VirtualPulseGenerator::readTimeData()
{
}

QVariant VirtualPulseGenerator::read(const int index, const BlackChirp::PulseSetting s)
{
    emit settingUpdate(index,s,d_config.setting(index,s));
    return d_config.setting(index,s);
}

double VirtualPulseGenerator::readRepRate()
{
    emit repRateUpdate(d_config.repRate());
    return d_config.repRate();
}

bool VirtualPulseGenerator::set(const int index, const BlackChirp::PulseSetting s, const QVariant val)
{
    d_config.set(index,s,val);
    read(index,s);
    return true;
}

bool VirtualPulseGenerator::setRepRate(double d)
{
    d_config.setRepRate(d);
    readRepRate();
    return true;
}
