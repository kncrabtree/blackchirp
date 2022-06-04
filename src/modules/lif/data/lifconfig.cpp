#include <modules/lif/data/lifconfig.h>

#include <modules/lif/data/liftrace.h>
#include <QFile>
#include <cmath>


LifConfig::LifConfig() : HeaderStorage(BC::Store::LIF::key)
{

}

bool LifConfig::isComplete() const
{
    if(d_enabled)
        return d_complete;

    return true;
}

double LifConfig::currentDelay() const
{
    return static_cast<double>(d_currentDelayIndex)*d_delayStepUs + d_delayStartUs;
}

double LifConfig::currentLaserPos() const
{
    return static_cast<double>(d_currentFrequencyIndex)*d_laserPosStep + d_laserPosStart;
}

QPair<double, double> LifConfig::delayRange() const
{
    return qMakePair(d_delayStartUs,d_delayStartUs + d_delayStepUs*d_delayPoints);
}

QPair<double, double> LifConfig::laserRange() const
{
    return qMakePair(d_laserPosStart,d_laserPosStart + d_laserPosStep*d_laserPosPoints);
}

int LifConfig::totalShots() const
{
    return d_delayPoints*d_laserPosPoints*d_shotsPerPoint;
}

int LifConfig::completedShots() const
{
    if(d_complete)
        return totalShots();

    int out = 0;
    for(int i=0; i < d_lifData.size(); i++)
    {
        for(int j=0; j < d_lifData.at(i).size(); j++)
            out += d_lifData.at(i).at(j).count();
    }

    return out;
}

QPair<int, int> LifConfig::lifGate() const
{
    return qMakePair(d_lifGateStartPoint,d_lifGateEndPoint);
}

QPair<int, int> LifConfig::refGate() const
{
    return qMakePair(d_refGateStartPoint,d_refGateEndPoint);
}

QVector<QVector<LifTrace> > LifConfig::lifData() const
{
    return d_lifData;
}

bool LifConfig::loadLifData(int num, const QString path)
{
    ///TODO

    Q_UNUSED(num)
    Q_UNUSED(path)
    return false;
}

bool LifConfig::addWaveform(const LifTrace t)
{
    //the boolean returned by this function tells if the point was incremented
    if(d_complete && d_completeMode == StopWhenComplete)
        return false;

    return(addTrace(t));

}

bool LifConfig::addTrace(const LifTrace t)
{
    if(d_currentDelayIndex >= d_lifData.size())
    {
        QVector<LifTrace> l;
        l.append(t);
        d_lifData.append(l);
    }
    else if(d_currentFrequencyIndex >= d_lifData.at(d_currentDelayIndex).size())
    {
        d_lifData[d_currentDelayIndex].append(t);
    }
    else
        d_lifData[d_currentDelayIndex][d_currentFrequencyIndex].add(t);

    //return true if we have enough shots for this point on this pass
    int c = d_lifData.at(d_currentDelayIndex).at(d_currentFrequencyIndex).count();
    bool inc = !(c % d_shotsPerPoint);
    if(inc)
        increment();
    return inc;

}

void LifConfig::increment()
{
    if(d_currentDelayIndex+1 >= d_delayPoints && d_currentFrequencyIndex+1 >= d_laserPosPoints)
        d_complete = true;

    if(d_order == LaserFirst)
    {
        if(d_currentFrequencyIndex+1 >= d_laserPosPoints)
            d_currentDelayIndex = (d_currentDelayIndex+1)%d_delayPoints;

        d_currentFrequencyIndex = (d_currentFrequencyIndex+1)%d_laserPosPoints;
    }
    else
    {
        if(d_currentDelayIndex+1 >= d_delayPoints)
            d_currentFrequencyIndex = (d_currentFrequencyIndex+1)%d_laserPosPoints;

        d_currentDelayIndex = (d_currentDelayIndex+1)%d_delayPoints;
    }
}

void LifConfig::storeValues()
{
    using namespace BC::Store::LIF;
    store(order,d_order);
    store(completeMode,d_completeMode);
    store(dStart,d_delayStartUs,BC::Unit::us);
    store(dStep,d_delayStepUs,BC::Unit::us);
    store(dPoints,d_delayPoints);
    store(lStart,d_laserPosStart);
    store(lStep,d_laserPosStep);
    store(lPoints,d_delayPoints);
    store(shotsPerPoint,d_shotsPerPoint);

}

void LifConfig::retrieveValues()
{
    using namespace BC::Store::LIF;
    d_order = retrieve(order,DelayFirst);
    d_completeMode = retrieve(completeMode,ContinueAveraging);
    d_delayStartUs = retrieve(dStart,0.0);
    d_delayStepUs = retrieve(dStep,0.0);
    d_delayPoints = retrieve(dPoints,0);
    d_laserPosStart = retrieve(lStart,0.0);
    d_laserPosStep = retrieve(lStep,0.0);
    d_laserPosPoints = retrieve(lPoints,0);
    d_shotsPerPoint = retrieve(shotsPerPoint,0);
}

void LifConfig::prepareChildren()
{
    addChild(&d_scopeConfig);
}
