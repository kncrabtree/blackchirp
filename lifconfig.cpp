#include "lifconfig.h"

#include "liftrace.h"
#include <QFile>
#include <math.h>


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

LifConfig::~LifConfig()
{

}

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

bool LifConfig::isValid() const
{
    return data->valid;
}

double LifConfig::currentDelay() const
{
    return static_cast<double>(data->currentDelayIndex)*data->delayStepUs + data->delayStartUs;
}

double LifConfig::currentFrequency() const
{
    return static_cast<double>(data->currentFrequencyIndex)*data->frequencyStep + data->frequencyStart;
}

QPair<double, double> LifConfig::delayRange() const
{
    return qMakePair(data->delayStartUs,data->delayEndUs);
}

double LifConfig::delayStep() const
{
    return data->delayStepUs;
}

QPair<double, double> LifConfig::frequencyRange() const
{
    return qMakePair(data->frequencyStart,data->frequencyEnd);
}

double LifConfig::frequencyStep() const
{
    return data->frequencyStep;
}

int LifConfig::numDelayPoints() const
{
    if(fabs(data->delayStartUs-data->delayEndUs) < data->delayStepUs)
        return 1;

    return static_cast<int>(floor(fabs((data->delayStartUs-data->delayEndUs)/data->delayStepUs)))+1;
}

int LifConfig::numFrequencyPoints() const
{
    if(fabs(data->frequencyStart-data->frequencyEnd) < data->frequencyStep)
        return 1;

    return static_cast<int>(floor(fabs((data->frequencyStart-data->frequencyEnd)/data->frequencyStep)))+1;
}

int LifConfig::shotsPerPoint() const
{
    return data->shotsPerPoint;
}

int LifConfig::totalShots() const
{
    return numDelayPoints()*numFrequencyPoints()*data->shotsPerPoint;
}

int LifConfig::completedShots() const
{
    if(data->complete)
        return totalShots();

    int out;
    if(data->order == BlackChirp::LifOrderFrequencyFirst)
    {
        out = data->currentDelayIndex*numFrequencyPoints()*data->shotsPerPoint;
        out += data->currentFrequencyIndex*data->shotsPerPoint;
        out += data->lifData.at(data->currentDelayIndex).at(data->currentFrequencyIndex).count;
    }
    else
    {
        out = data->currentFrequencyIndex*numDelayPoints()*data->shotsPerPoint;
        out += data->currentDelayIndex*data->shotsPerPoint;
        out += data->lifData.at(data->currentDelayIndex).at(data->currentFrequencyIndex).count;
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

QVector<QPointF> LifConfig::timeSlice(int frequencyIndex) const
{
    QVector<QPointF> out;
    out.resize(data->lifData.size());

    if(data->lifData.isEmpty())
        return out;

    if(frequencyIndex >= numFrequencyPoints())
        return out;

    if(data->delayStepUs > 0.0)
    {
        for(int i=0; i<data->lifData.size()-1; i++)
        {
            out[i].setX(data->delayStartUs + static_cast<double>(i)*data->delayStepUs);
            out[i].setY(data->lifData.at(i).at(frequencyIndex).mean);
        }
        out[data->lifData.size()-1].setX(data->delayEndUs);
        out[data->lifData.size()-1].setY(data->lifData.at(data->lifData.size()-1).at(frequencyIndex).mean);
    }
    else
    {
        out[0].setX(data->delayEndUs);
        out[0].setY(data->lifData.at(0).at(frequencyIndex).mean);
        for(int i=data->lifData.size()-1; i>0; i--)
        {
            out[i].setX(data->delayStartUs + static_cast<double>(i)*data->delayStepUs);
            out[i].setY(data->lifData.at(i).at(frequencyIndex).mean);
        }
    }

    return out;
}

QVector<QPointF> LifConfig::spectrum(int delayIndex) const
{
    QVector<QPointF> out;
    if(delayIndex >= data->lifData.size())
        return out;

    out.resize(data->lifData.at(delayIndex).size());

    if(data->frequencyStep > 0.0)
    {
        for(int i=0; i<data->lifData.at(delayIndex).size()-1; i++)
        {
            out[i].setX(data->frequencyStart + static_cast<double>(i)*data->frequencyStep);
            out[i].setY(data->lifData.at(delayIndex).at(i).mean);
        }
        out[data->lifData.at(delayIndex).size()-1].setX(data->delayEndUs);
        out[data->lifData.at(delayIndex).size()-1].setY(data->lifData.at(delayIndex).at(data->lifData.at(delayIndex).size()-1).mean);
    }
    else
    {
        out[0].setX(data->delayEndUs);
        out[0].setY(data->lifData.at(delayIndex).at(0).mean);
        for(int i=data->lifData.size()-1; i>0; i--)
        {
            out[i].setX(data->frequencyStart + static_cast<double>(i)*data->frequencyStep);
            out[i].setY(data->lifData.at(delayIndex).at(i).mean);
        }
    }

    return out;
}

QList<QVector<BlackChirp::LifPoint> > LifConfig::lifData() const
{
    return data->lifData;
}

void LifConfig::setEnabled(bool en)
{
//    if(!data->valid)
//        return;

    data->enabled = en;
}

bool LifConfig::validate()
{
    data->valid = false;

    if(numDelayPoints() < 1 || numFrequencyPoints() < 1)
        return false;

    data->valid = true;
    return true;
}

bool LifConfig::allocateMemory()
{
    if(!data->valid)
        return false;

    if(!data->enabled)
        return false;

    //allocate memory for storage
    for(int i=0; i<numDelayPoints(); i++)
    {
        QVector<BlackChirp::LifPoint> d;
        d.resize(numFrequencyPoints());
        data->lifData.append(d);
    }

    //set signs for steps
    if(numDelayPoints() > 1)
    {
        if(data->delayStartUs > data->delayEndUs)
            data->delayStepUs = -data->delayStepUs;
    }

    if(numFrequencyPoints() > 1)
    {
        if(data->frequencyStart > data->frequencyEnd)
            data->frequencyStep = -data->frequencyStep;
    }

    data->memAllocated = true;
    return true;
}

void LifConfig::setLifGate(int start, int end)
{
    data->lifGateStartPoint = start;
    data->lifGateEndPoint = end;
}

void LifConfig::setRefGate(int start, int end)
{
    data->scopeConfig.refEnabled = true;
    data->refGateStartPoint=start;
    data->refGateEndPoint = end;
}

void LifConfig::setDelayParameters(double start, double stop, double step)
{
    data->delayStartUs = start;
    data->delayEndUs = stop;
    data->delayStepUs = step;
}

void LifConfig::setFrequencyParameters(double start, double stop, double step)
{
    data->frequencyStart = start;
    data->frequencyEnd = stop;
    data->frequencyStep = step;
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
    if(numDelayPoints() > 1)
    {
        out.insert(prefix+QString("DelayStart"),
                   qMakePair(QString::number(data->delayStartUs,'f',3),QString::fromUtf16(u"µs")));
        out.insert(prefix+QString("DelayStop"),
                   qMakePair(QString::number(data->delayEndUs,'f',3),QString::fromUtf16(u"µs")));
        out.insert(prefix+QString("DelayStep"),
                   qMakePair(QString::number(data->delayStepUs,'f',3),QString::fromUtf16(u"µs")));
    }
    else
        out.insert(prefix+QString("Delay"),
                   qMakePair(QString::number(data->delayStartUs,'f',3),QString::fromUtf16(u"µs")));
    if(numFrequencyPoints() > 1)
    {
        out.insert(prefix+QString("FrequencyStart"),
                   qMakePair(QString::number(data->frequencyStart,'f',3),QString("1/cm")));
        out.insert(prefix+QString("FrequencyStop"),
                   qMakePair(QString::number(data->frequencyEnd,'f',3),QString("1/cm")));
        out.insert(prefix+QString("FrequencyStep"),
                   qMakePair(QString::number(data->frequencyStep,'f',3),QString("1/cm")));
    }
    else
        out.insert(prefix+QString("Frequency"),
                   qMakePair(QString::number(data->frequencyStart,'f',3),QString("1/cm")));

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
        if(key.endsWith(QString("DelayStop")))
            data->delayEndUs = val.toDouble();
        if(key.endsWith(QString("DelayStep")))
            data->delayStepUs = val.toDouble();
        if(key.endsWith(QString("FrequencyStart")) || key.endsWith(QString("Frequency")))
            data->frequencyStart = val.toDouble();
        if(key.endsWith(QString("FrequencyStop")))
            data->frequencyEnd = val.toDouble();
        if(key.endsWith(QString("FrequencyStep")))
            data->frequencyStep = val.toDouble();
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
                data->scopeConfig.byteOrder = QDataStream::LittleEndian;
            else
                data->scopeConfig.byteOrder = QDataStream::BigEndian;
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
        if(magic.endsWith("v1.0"))
        {
            QList<QVector<BlackChirp::LifPoint>> l;
            d >> l;
            data->lifData = l;
        }

        lif.close();
        return true;
    }

    return false;
}

QPair<QPoint, BlackChirp::LifPoint> LifConfig::lastUpdatedLifPoint() const
{
    if(data->lastUpdatedPoint.x() < data->lifData.size())
    {
        if(data->lastUpdatedPoint.y() < data->lifData.at(data->lastUpdatedPoint.x()).size())
        {
            return qMakePair(data->lastUpdatedPoint,data->lifData.at(data->lastUpdatedPoint.x()).at(data->lastUpdatedPoint.y()));
//            BlackChirp::LifPoint p;
//            p.mean = static_cast<double>(qrand() % 1000)/100.0;
//            return qMakePair(data->lastUpdatedPoint,p);
        }
    }

    return qMakePair(QPoint(-1,-1),BlackChirp::LifPoint());

}

bool LifConfig::writeLifFile(int num) const
{
    QFile lif(BlackChirp::getExptFile(num,BlackChirp::LifFile));
    if(lif.open(QIODevice::WriteOnly))
    {
        QDataStream d(&lif);
        d << QByteArray("BCLIFv1.0");
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

    double d;
    if(data->scopeConfig.refEnabled)
        d = t.integrate(data->lifGateStartPoint,data->lifGateEndPoint,data->refGateStartPoint,data->refGateEndPoint);
    else
        d = t.integrate(data->lifGateStartPoint,data->lifGateEndPoint);

    bool inc = addPoint(d);
    if(inc)
        increment();

    return inc;
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
    s.setValue(QString("delayEnd"),data->delayEndUs);
    s.setValue(QString("delayStep"),data->delayStepUs);
    s.setValue(QString("laserSingle"),(numFrequencyPoints() == 1));
    s.setValue(QString("laserStart"),data->frequencyStart);
    s.setValue(QString("laserEnd"),data->frequencyEnd);
    s.setValue(QString("laserStep"),data->frequencyStep);

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
    out.setDelayParameters(s.value(QString("delayStart"),1000.0).toDouble(),s.value(QString("delayEnd"),1100.0).toDouble(),s.value(QString("delayStep"),10.0).toDouble());
    out.setFrequencyParameters(s.value(QString("laserStart"),15000.0).toDouble(),s.value(QString("laserEnd"),15100.0).toDouble(),s.value(QString("laserStep"),5.0).toDouble());

    out.validate();
    return out;
}

bool LifConfig::addPoint(const double d)
{
    int i = data->currentDelayIndex;
    int j = data->currentFrequencyIndex;

    data->lifData[i][j].count++;
    double delta = d - data->lifData[i][j].mean;
    data->lifData[i][j].mean += delta/static_cast<double>(data->lifData[i][j].count);
    data->lifData[i][j].sumsq += delta*(d - data->lifData[i][j].mean);

    data->lastUpdatedPoint.setX(i);
    data->lastUpdatedPoint.setY(j);

    //return true if we've collected shotsPerPoint shots on this pass
    return !(data->lifData[i][j].count % data->shotsPerPoint);
}

void LifConfig::increment()
{
    if(data->currentDelayIndex+1 >= numDelayPoints() && data->currentFrequencyIndex+1 >= numFrequencyPoints())
        data->complete = true;

    if(data->order == BlackChirp::LifOrderFrequencyFirst)
    {
        if(data->currentFrequencyIndex+1 >= numFrequencyPoints())
            data->currentDelayIndex = (data->currentDelayIndex+1)%numDelayPoints();

        data->currentFrequencyIndex = (data->currentFrequencyIndex+1)%numFrequencyPoints();
    }
    else
    {
        if(data->currentDelayIndex+1 >= numDelayPoints())
            data->currentFrequencyIndex = (data->currentFrequencyIndex+1)%numFrequencyPoints();

        data->currentDelayIndex = (data->currentDelayIndex+1)%numDelayPoints();
    }
}



QDataStream &operator<<(QDataStream &stream, const BlackChirp::LifPoint &pt)
{
    stream << pt.count << pt.mean << pt.sumsq;
    return stream;
}


QDataStream &operator>>(QDataStream &stream, BlackChirp::LifPoint &pt)
{
    stream >> pt.count >> pt.mean >> pt.sumsq;
    return stream;
}
