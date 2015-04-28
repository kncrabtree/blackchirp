#include "lifconfig.h"


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
    return data->complete;
}

double LifConfig::currentDelay() const
{
    return static_cast<double>(data->currentDelayIndex)*data->delayStepUs + data->delayStartUs;
}

double LifConfig::currentFrequency() const
{
    return static_cast<double>(data->currentFrequencyIndex)*data->frequencyStep + data->frequencyStart;
}

int LifConfig::numDelayPoints() const
{
    if(fabs(data->delayStartUs-data->delayEndUs) < data->delayStepUs)
        return 1;

    return static_cast<int>(floor(fabs((data->delayStartUs-data->delayEndUs)/data->delayStepUs)))+2;
}

int LifConfig::numFrequencyPoints() const
{
    if(fabs(data->frequencyStart-data->frequencyEnd) < data->frequencyStep)
        return 1;

    return static_cast<int>(floor(fabs((data->frequencyStart-data->frequencyEnd)/data->frequencyStep)))+2;
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
    if(data->order == LifConfig::DelayFirst)
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

bool LifConfig::setEnabled()
{
    if(!data->valid)
        return false;

    //allocate memory for storage
    for(int i=0; i<numDelayPoints(); i++)
    {
        QVector<LifConfig::LifPoint> d;
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

    data->enabled = true;
    return true;
}

bool LifConfig::validate()
{
    data->valid = false;

    if(numDelayPoints() < 1 || numFrequencyPoints() < 1)
        return false;

    if(data->lifGateEndPoint < 0 || data->lifGateStartPoint < 0)
        return false;

    data->valid = true;
    return true;
}

void LifConfig::setLifGate(int start, int end)
{
    data->lifGateStartPoint = start;
    data->lifGateEndPoint = end;
}

void LifConfig::setRefGate(int start, int end)
{
    data->refEnabled = true;
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

void LifConfig::setOrder(LifConfig::ScanOrder o)
{
    data->order = o;
}

void LifConfig::setShotsPerPoint(int pts)
{
    data->shotsPerPoint = pts;
}

LifTrace LifConfig::parseWaveform(const QByteArray b) const
{
    if(!data->refEnabled)
        return LifTrace(b,data->scopeConfig.byteOrder,data->scopeConfig.bytesPerPoint,data->scopeConfig.recordLength,
                        data->scopeConfig.xIncr,data->scopeConfig.yMult1);
    else
        return LifTrace(b,data->scopeConfig.byteOrder,data->scopeConfig.bytesPerPoint,data->scopeConfig.recordLength,
                        data->scopeConfig.xIncr,data->scopeConfig.yMult1,true,data->scopeConfig.yMult2);
}

QMap<QString, QPair<QVariant, QString> > LifConfig::headerMap() const
{
    QMap<QString,QPair<QVariant,QString> > out;
    QString empty = QString("");
    QString prefix = QString("LifScope");
    QString so = (data->order == DelayFirst ? QString("DelayFirst") : QString("FrequencyFirst"));


    out.insert(prefix+QString("ScanOrder"),qMakePair(so,empty));
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

    if(data->refEnabled)
    {
        out.insert(prefix+QString("RefGateStart"),qMakePair(data->refGateStartPoint,empty));
        out.insert(prefix+QString("RefGateStop"),qMakePair(data->refGateEndPoint,empty));
    }

    out.unite(data->scopeConfig.headerMap());

    return out;
}

QPair<QPoint, LifConfig::LifPoint> LifConfig::lastUpdatedLifPoint() const
{
    if(data->lastUpdatedPoint.x() < data->lifData.size())
    {
        if(data->lastUpdatedPoint.y() < data->lifData.at(data->lastUpdatedPoint.x()).size())
            return qMakePair(data->lastUpdatedPoint,data->lifData.at(data->lastUpdatedPoint.x()).at(data->lastUpdatedPoint.y()));
    }

    return qMakePair(QPoint(-1,-1),LifPoint());

}

bool LifConfig::addWaveform(const LifTrace t)
{
    //the boolean returned by this function tells if the point was incremented

    //do trapezoidal integration in integer/point space.
    //each segment has a width of 1 unit, and the area is (y_i + y_{i+1})/2
    //(think of it as end points have weight of 1, middle points have weight 2)
    //add up all y_i + y_{i+1}, then divide by 2 at the end

    //convert to double using scope scaling, and add point
    double d;
    if(data->refEnabled)
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

    if(data->order == LifConfig::DelayFirst)
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

