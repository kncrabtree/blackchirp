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
    return data->lifData.size();
}

int LifConfig::numFrequencyPoints() const
{
    if(data->lifData.isEmpty())
        return 0;

    return data->lifData.at(0).size();
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

void LifConfig::setEnabled()
{
    data->enabled = true;
}

bool LifConfig::addWaveform(const QByteArray b)
{
    //the boolean returned by this function tells if the point was incremented

    //do trapezoidal integration in integer/point space.
    //each segment has a width of 1 unit, and the area is (y_i + y_{i+1})/2
    //(think of it as end points have weight of 1, middle points have weight 2)
    //add up all y_i + y_{i+1}, then divide by 2 at the end
    qint64 integral = 0;

    int stride = 1;
    if(data->scopeConfig.bytesPerPoint != 1)
        stride = 2;
    for(int i=data->gateStartPoint*stride; i+stride<data->gateEndPoint*stride; i+=stride)
    {
        if(data->scopeConfig.bytesPerPoint == 1)
            integral += static_cast<qint64>(b.at(i)) + static_cast<qint64>(b.at(i+1));
        else
        {
            qint16 dat1, dat2;
            if(data->scopeConfig.byteOrder == QDataStream::LittleEndian)
            {
                dat1 = (b.at(i+1) << 8) | (b.at(i) & 0xff);
                dat2 = (b.at(i+3) << 8) | (b.at(i+2) & 0xff);
            }
            else
            {
                dat1 = (b.at(i) << 8) | (b.at(i+1) & 0xff);
                dat2 = (b.at(i+2) << 8) | (b.at(i+3) & 0xff);
            }
            integral += static_cast<qint64>(dat1) + static_cast<qint64>(dat2);
        }
    }

    //convert to double using scope scaling, and add point
    double d = static_cast<double>(integral)*data->scopeConfig.yMult/2.0;
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

