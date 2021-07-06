#include <modules/lif/data/lifconfig.h>

#include <modules/lif/data/liftrace.h>
#include <QFile>
#include <cmath>


LifConfig::LifConfig() : data(new LifConfigData)
{

}

LifConfig::LifConfig(const LifConfig &rhs) : data(rhs.data)
{

}

LifConfig &LifConfig::operator=(const LifConfig &rhs)
{
    if (this != &rhs)
        data.operator=(rhs.data);
    return *this;
}

LifConfig::~LifConfig() = default;

bool LifConfig::isEnabled() const
{
    return data->enabled;
}

bool LifConfig::isComplete() const
{
    if(data->enabled)
        return data->complete;

    return true;
}

double LifConfig::currentDelay() const
{
    return static_cast<double>(data->currentDelayIndex)*data->delayStepUs + data->delayStartUs;
}

double LifConfig::currentLaserPos() const
{
    return static_cast<double>(data->currentFrequencyIndex)*data->laserPosStep + data->laserPosStart;
}

QPair<double, double> LifConfig::delayRange() const
{
    return qMakePair(data->delayStartUs,data->delayStartUs + data->delayStepUs*data->delayPoints);
}

double LifConfig::delayStep() const
{
    return data->delayStepUs;
}

QPair<double, double> LifConfig::laserRange() const
{
    return qMakePair(data->laserPosStart,data->laserPosStart + data->laserPosStep*data->laserPosPoints);
}

double LifConfig::laserStep() const
{
    return data->laserPosStep;
}

int LifConfig::numDelayPoints() const
{
    return data->delayPoints;
}

int LifConfig::numLaserPoints() const
{
    return data->laserPosPoints;
}

int LifConfig::shotsPerPoint() const
{
    return data->shotsPerPoint;
}

int LifConfig::totalShots() const
{
    return numDelayPoints()*numLaserPoints()*data->shotsPerPoint;
}

int LifConfig::completedShots() const
{
    if(data->complete)
        return totalShots();

    int out = 0;
    for(int i=0; i < data->lifData.size(); i++)
    {
        for(int j=0; j < data->lifData.at(i).size(); j++)
            out += data->lifData.at(i).at(j).count();
    }

    return out;
}

BlackChirp::LifScopeConfig LifConfig::scopeConfig() const
{
    return data->scopeConfig;
}

BlackChirp::LifScanOrder LifConfig::order() const
{
    return data->order;
}

BlackChirp::LifCompleteMode LifConfig::completeMode() const
{
    return data->completeMode;
}

QPair<int, int> LifConfig::lifGate() const
{
    return qMakePair(data->lifGateStartPoint,data->lifGateEndPoint);
}

QPair<int, int> LifConfig::refGate() const
{
    return qMakePair(data->refGateStartPoint,data->refGateEndPoint);
}

QList<QList<LifTrace> > LifConfig::lifData() const
{
    return data->lifData;
}

void LifConfig::setEnabled(bool en)
{
    data->enabled = en;
}

void LifConfig::setLifGate(int start, int end)
{
    data->lifGateStartPoint = start;
    data->lifGateEndPoint = end;
}

void LifConfig::setLifGate(const QPair<int, int> p)
{
    data->lifGateStartPoint = p.first;
    data->lifGateEndPoint = p.second;
}

void LifConfig::setRefGate(int start, int end)
{
    data->scopeConfig.refEnabled = true;
    data->refGateStartPoint=start;
    data->refGateEndPoint = end;
}

void LifConfig::setRefGate(const QPair<int, int> p)
{
    data->scopeConfig.refEnabled = true;
    data->refGateStartPoint = p.first;
    data->refGateEndPoint = p.second;
}

void LifConfig::setDelayParameters(double start, double step, int count)
{
    data->delayStartUs = start;
    data->delayStepUs = step;
    data->delayPoints = count;
}

void LifConfig::setLaserParameters(double start, double step, int count)
{
    data->laserPosStart = start;
    data->laserPosStep = step;
    data->laserPosPoints = count;
}

void LifConfig::setOrder(BlackChirp::LifScanOrder o)
{
    data->order = o;
}

void LifConfig::setCompleteMode(BlackChirp::LifCompleteMode mode)
{
    data->completeMode = mode;
}

void LifConfig::setScopeConfig(BlackChirp::LifScopeConfig c)
{
    data->scopeConfig = c;
}

void LifConfig::setShotsPerPoint(int pts)
{
    data->shotsPerPoint = pts;
}

QMap<QString, QPair<QVariant, QString> > LifConfig::headerMap() const
{
    QMap<QString,QPair<QVariant,QString> > out;
    QString empty = QString("");
    QString prefix = QString("LifConfig");
    QString so = (data->order == BlackChirp::LifOrderDelayFirst ?
                      QString("DelayFirst") : QString("FrequencyFirst"));
    QString comp = (data->completeMode == BlackChirp::LifStopWhenComplete ?
                        QString("Stop") : QString("Continue"));

    out.insert(prefix+QString("Enabled"),qMakePair(isEnabled(),empty));
    if(!isEnabled())
        return out;

    out.insert(prefix+QString("ScanOrder"),qMakePair(so,empty));
    out.insert(prefix+QString("CompleteBehavior"),qMakePair(comp,empty));
    out.insert(prefix+QString("DelayStart"),
               qMakePair(QString::number(data->delayStartUs,'f',3),QString::fromUtf16(u"µs")));
    out.insert(prefix+QString("DelayPoints"),
               qMakePair(QString::number(data->delayPoints),QString("")));
    out.insert(prefix+QString("DelayStep"),
               qMakePair(QString::number(data->delayStepUs,'f',3),QString::fromUtf16(u"µs")));
    out.insert(prefix+QString("FrequencyStart"),
               qMakePair(QString::number(data->laserPosStart,'f',3),QString("1/cm")));
    out.insert(prefix+QString("FrequencyPoints"),
               qMakePair(QString::number(data->laserPosPoints),QString("")));
    out.insert(prefix+QString("FrequencyStep"),
               qMakePair(QString::number(data->laserPosStep,'f',3),QString("1/cm")));

    out.insert(prefix+QString("ShotsPerPoint"),qMakePair(data->shotsPerPoint,empty));
    out.insert(prefix+QString("LifGateStart"),qMakePair(data->lifGateStartPoint,empty));
    out.insert(prefix+QString("LifGateStop"),qMakePair(data->lifGateEndPoint,empty));
    if(data->scopeConfig.refEnabled)
    {
        out.insert(prefix+QString("RefGateStart"),qMakePair(data->refGateStartPoint,empty));
        out.insert(prefix+QString("RefGateStop"),qMakePair(data->refGateEndPoint,empty));
    }

    out.unite(data->scopeConfig.headerMap());

    return out;
}

void LifConfig::parseLine(QString key, QVariant val)
{
    if(key.startsWith(QString("LifConfig")))
    {
        if(key.endsWith(QString("Enabled")))
            data->enabled = val.toBool();
        if(key.endsWith(QString("ScanOrder")))
        {
            if(val.toString().contains(QString("Delay")))
                data->order = BlackChirp::LifOrderDelayFirst;
            else
                data->order = BlackChirp::LifOrderFrequencyFirst;
        }
        if(key.endsWith(QString("CompleteBehavior")))
        {
            if(val.toString().contains(QString("Stop")))
                data->completeMode = BlackChirp::LifStopWhenComplete;
            else
                data->completeMode = BlackChirp::LifContinueUntilExperimentComplete;
        }
        if(key.endsWith(QString("DelayStart")) || key.endsWith(QString("Delay")))
            data->delayStartUs = val.toDouble();
        if(key.endsWith(QString("DelayPoints")))
            data->delayPoints = val.toInt();
        if(key.endsWith(QString("DelayStep")))
            data->delayStepUs = val.toDouble();
        if(key.endsWith(QString("FrequencyStart")) || key.endsWith(QString("Frequency")))
            data->laserPosStart = val.toDouble();
        if(key.endsWith(QString("FrequencyPoints")))
            data->laserPosPoints = val.toInt();
        if(key.endsWith(QString("FrequencyStep")))
            data->laserPosStep = val.toDouble();
        if(key.endsWith(QString("ShotsPerPoint")))
            data->shotsPerPoint = val.toInt();
        if(key.endsWith(QString("LifGateStart")))
            data->lifGateStartPoint = val.toInt();
        if(key.endsWith(QString("LifGateStop")))
            data->lifGateEndPoint = val.toInt();
        if(key.endsWith(QString("RefGateStart")))
            data->refGateStartPoint = val.toInt();
        if(key.endsWith(QString("RefGateStop")))
            data->refGateEndPoint = val.toInt();
    }
    else if(key.startsWith(QString("LifScope")))
    {
        if(key.endsWith(QString("LifVerticalScale")))
            data->scopeConfig.vScale1 = val.toDouble();
        if(key.endsWith(QString("RefVerticalScale")))
            data->scopeConfig.vScale2 = val.toDouble();
        if(key.endsWith(QString("TriggerSlope")))
        {
            if(val.toString().contains(QString("Rising")))
                data->scopeConfig.slope = BlackChirp::RisingEdge;
            else
                data->scopeConfig.slope = BlackChirp::FallingEdge;
        }
        if(key.endsWith(QString("SampleRate")))
            data->scopeConfig.sampleRate = val.toDouble()*1e9;
        if(key.endsWith(QString("RecordLength")))
            data->scopeConfig.recordLength = val.toInt();
        if(key.endsWith(QString("BytesPerPoint")))
            data->scopeConfig.bytesPerPoint = val.toInt();
        if(key.endsWith(QString("ByteOrder")))
        {
            if(val.toString().contains(QString("Little")))
                data->scopeConfig.byteOrder = DigitizerConfig::LittleEndian;
            else
                data->scopeConfig.byteOrder = DigitizerConfig::BigEndian;
        }

    }

}

bool LifConfig::loadLifData(int num, const QString path)
{
    QFile lif(BlackChirp::getExptFile(num,BlackChirp::LifFile,path));
    if(lif.open(QIODevice::ReadOnly))
    {
        QDataStream d(&lif);
        QByteArray magic;
        d >> magic;
        if(!magic.startsWith("BCLIF"))
        {
            lif.close();
            return false;
        }

        bool success = false;
        if(magic.endsWith("v1.1"))
        {
            QList<QList<LifTrace>> l;
            d >> l;
            data->lifData = l;
            success = true;
        }

        lif.close();
        return success;
    }

    return false;
}

bool LifConfig::writeLifFile(int num) const
{
    QFile lif(BlackChirp::getExptFile(num,BlackChirp::LifFile));
    if(lif.open(QIODevice::WriteOnly))
    {
        QDataStream d(&lif);
        d << QByteArray("BCLIFv1.1");
        d << data->lifData;
        lif.close();
        return true;
    }
    else
        return false;
}

bool LifConfig::addWaveform(const LifTrace t)
{
    //the boolean returned by this function tells if the point was incremented
    if(data->complete && data->completeMode == BlackChirp::LifStopWhenComplete)
        return false;

    return(addTrace(t));

}

void LifConfig::saveToSettings() const
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());

    //scope settings and graph settings are saved on the fly, so don't worry about those
    s.beginGroup(QString("lastLifConfig"));
    s.setValue(QString("scanOrder"),static_cast<int>(order()));
    s.setValue(QString("completeMode"),static_cast<int>(completeMode()));
    s.setValue(QString("delaySingle"),(numDelayPoints() == 1));
    s.setValue(QString("delayStart"),data->delayStartUs);
    s.setValue(QString("delayPoints"),data->delayPoints);
    s.setValue(QString("delayStep"),data->delayStepUs);
    s.setValue(QString("laserSingle"),(numLaserPoints() == 1));
    s.setValue(QString("laserStart"),data->laserPosStart);
    s.setValue(QString("laserPoints"),data->laserPosPoints);
    s.setValue(QString("laserStep"),data->laserPosStep);

    s.endGroup();
    s.sync();
}

LifConfig LifConfig::loadFromSettings()
{
    //scope settings have to come from the control widget on UI or wizard
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(QString("lastLifConfig"));

    LifConfig out;
    out.setCompleteMode(static_cast<BlackChirp::LifCompleteMode>(s.value(QString("completeMode"),0).toInt()));
    out.setOrder(static_cast<BlackChirp::LifScanOrder>(s.value(QString("scanOrder"),0).toInt()));
    out.setDelayParameters(s.value(QString("delayStart"),1000.0).toDouble(),s.value(QString("delayStep"),10.0).toDouble(),s.value(QString("delayPoints"),1).toInt());
    out.setLaserParameters(s.value(QString("laserStart"),15000.0).toDouble(),s.value(QString("laserStep"),5.0).toDouble(),s.value(QString("laserPoints"),1).toInt());

    return out;
}

bool LifConfig::addTrace(const LifTrace t)
{
    if(data->currentDelayIndex >= data->lifData.size())
    {
        QList<LifTrace> l;
        l.append(t);
        data->lifData.append(l);
    }
    else if(data->currentFrequencyIndex >= data->lifData.at(data->currentDelayIndex).size())
    {
        data->lifData[data->currentDelayIndex].append(t);
    }
    else
        data->lifData[data->currentDelayIndex][data->currentFrequencyIndex].add(t);

    //return true if we have enough shots for this point on this pass
    int c = data->lifData.at(data->currentDelayIndex).at(data->currentFrequencyIndex).count();
    bool inc = !(c % data->shotsPerPoint);
    if(inc)
        increment();
    return inc;

}

void LifConfig::increment()
{
    if(data->currentDelayIndex+1 >= numDelayPoints() && data->currentFrequencyIndex+1 >= numLaserPoints())
        data->complete = true;

    if(data->order == BlackChirp::LifOrderFrequencyFirst)
    {
        if(data->currentFrequencyIndex+1 >= numLaserPoints())
            data->currentDelayIndex = (data->currentDelayIndex+1)%numDelayPoints();

        data->currentFrequencyIndex = (data->currentFrequencyIndex+1)%numLaserPoints();
    }
    else
    {
        if(data->currentDelayIndex+1 >= numDelayPoints())
            data->currentFrequencyIndex = (data->currentFrequencyIndex+1)%numLaserPoints();

        data->currentDelayIndex = (data->currentDelayIndex+1)%numDelayPoints();
    }
}


