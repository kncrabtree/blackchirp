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
    return static_cast<double>(d_currentLaserIndex)*d_laserPosStep + d_laserPosStart;
}

QPair<double, double> LifConfig::delayRange() const
{
    return qMakePair(d_delayStartUs,d_delayStartUs + d_delayStepUs*d_delayPoints);
}

QPair<double, double> LifConfig::laserRange() const
{
    return qMakePair(d_laserPosStart,d_laserPosStart + d_laserPosStep*d_laserPosPoints);
}

int LifConfig::targetShots() const
{
    return d_delayPoints*d_laserPosPoints*d_shotsPerPoint;
}

int LifConfig::completedShots() const
{
    return ps_storage->completedShots();
}

QPair<int, int> LifConfig::lifGate() const
{
    return qMakePair(d_lifGateStartPoint,d_lifGateEndPoint);
}

QPair<int, int> LifConfig::refGate() const
{
    return qMakePair(d_refGateStartPoint,d_refGateEndPoint);
}

void LifConfig::addWaveform(const QByteArray d)
{
    //the boolean returned by this function tells if the point was incremented
    if(d_complete && d_completeMode == StopWhenComplete)
        return;

    LifTrace t(d_scopeConfig,d,d_currentDelayIndex,d_currentLaserIndex);
    ps_storage->addTrace(t);
}

void LifConfig::loadLifData()
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
    store(lifGateStart,d_lifGateStartPoint);
    store(lifGateEnd,d_lifGateEndPoint);
    store(refGateStart,d_refGateStartPoint);
    store(refGateEnd,d_refGateEndPoint);

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
    d_lifGateStartPoint = retrieve(lifGateStart,-1);
    d_lifGateEndPoint = retrieve(lifGateEnd,-1);
    d_refGateStartPoint = retrieve(refGateStart,-1);
    d_refGateEndPoint = retrieve(refGateEnd,-1);
}

void LifConfig::prepareChildren()
{
    addChild(&d_scopeConfig);
}


bool LifConfig::initialize()
{
    ps_storage = std::make_shared<LifStorage>(d_delayPoints,d_laserPosPoints,d_number,d_path);
    ps_storage->start();
    return true;
}

bool LifConfig::advance()
{
    //return true if we have enough shots for this point on this pass
    int c = ps_storage->currentTraceShots();
    int target = d_shotsPerPoint*(d_completedSweeps+1);

    bool inc = (c>=target);
    if(inc)
    {
        d_processingPaused = true;
        if(d_currentDelayIndex+1 >= d_delayPoints && d_currentLaserIndex+1 >= d_laserPosPoints)
        {
            d_completedSweeps++;
            d_complete = true;
        }

        if(d_order == LaserFirst)
        {
            if(d_currentLaserIndex+1 >= d_laserPosPoints)
                d_currentDelayIndex = (d_currentDelayIndex+1)%d_delayPoints;

            d_currentLaserIndex = (d_currentLaserIndex+1)%d_laserPosPoints;
        }
        else
        {
            if(d_currentDelayIndex+1 >= d_delayPoints)
                d_currentLaserIndex = (d_currentLaserIndex+1)%d_laserPosPoints;

            d_currentDelayIndex = (d_currentDelayIndex+1)%d_delayPoints;
        }
        ps_storage->advance();
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


void LifConfig::cleanupAndSave()
{
    ps_storage->finish();
    ps_storage->save();
}
