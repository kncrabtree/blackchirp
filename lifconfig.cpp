#include "lifconfig.h"

#include "liftrace.h"


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

QPair<double, double> LifConfig::frequencyRange() const
{
    return qMakePair(data->frequencyStart,data->frequencyEnd);
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

void LifConfig::setEnabled()
{
    if(!data->valid)
        return;

    data->enabled = true;
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

QPair<QPoint, BlackChirp::LifPoint> LifConfig::lastUpdatedLifPoint() const
{
    if(data->lastUpdatedPoint.x() < data->lifData.size())
    {
        if(data->lastUpdatedPoint.y() < data->lifData.at(data->lastUpdatedPoint.x()).size())
        {
//            return qMakePair(data->lastUpdatedPoint,data->lifData.at(data->lastUpdatedPoint.x()).at(data->lastUpdatedPoint.y()));
            BlackChirp::LifPoint p;
            p.mean = static_cast<double>(qrand() % 1000)/100.0;
            return qMakePair(data->lastUpdatedPoint,p);
        }
    }

    return qMakePair(QPoint(-1,-1),BlackChirp::LifPoint());

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

