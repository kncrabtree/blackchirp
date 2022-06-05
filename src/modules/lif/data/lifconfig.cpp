#include <modules/lif/data/lifconfig.h>

#include <modules/lif/data/liftrace.h>
#include <QFile>
#include <cmath>


LifConfig::LifConfig() : HeaderStorage(BC::Store::LIF::key)
{

}

bool LifConfig::isComplete() const
{
    return d_complete;
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

void LifConfig::addWaveform(const QByteArray d)
{
    //the boolean returned by this function tells if the point was incremented
    if(d_complete && d_completeMode == StopWhenComplete)
        return;

    return addTrace(d);

}

void LifConfig::addTrace(const QByteArray d)
{
    LifTrace t(d_scopeConfig,d);
    if(d_currentDelayIndex >= d_lifData.size())
    {
        QVector<LifTrace> l;
        l.append(std::move(t));
        d_lifData.append(l);
    }
    else if(d_currentFrequencyIndex >= d_lifData.at(d_currentDelayIndex).size())
    {
        d_lifData[d_currentDelayIndex].append(std::move(t));
    }
    else
        d_lifData[d_currentDelayIndex][d_currentFrequencyIndex].add(t);

}

void LifConfig::increment()
{

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


bool LifConfig::initialize()
{
    return true;
}

bool LifConfig::advance()
{
    //return true if we have enough shots for this point on this pass
    int c = d_lifData.at(d_currentDelayIndex).at(d_currentFrequencyIndex).count();
    int target = d_shotsPerPoint*(d_completedSweeps+1);

    bool inc = c>=target;
    if(inc)
    {
        d_processingPaused = true;
        if(d_currentDelayIndex+1 >= d_delayPoints && d_currentFrequencyIndex+1 >= d_laserPosPoints)
        {
            d_completedSweeps++;
            d_complete = true;
        }

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
    return inc;
}

void LifConfig::hwReady()
{
    d_processingPaused = false;
}

int LifConfig::perMilComplete() const
{
    auto i = completedShots();
    return qBound(0,(1000*i)/(d_shotsPerPoint*d_delayPoints*d_laserPosPoints),1000);
}

bool LifConfig::indefinite() const
{
    if(d_completeMode == ContinueAveraging)
        return perMilComplete() >= 1000;

    return false;
}

bool LifConfig::abort()
{
    return false;
}

QString LifConfig::objectiveKey() const
{
    return BC::Config::Exp::lifType;
}
