#include <modules/lif/hardware/liflaser/liflaser.h>
#include <modules/lif/hardware/lifdigitizer/lifscope.h>
#include <modules/lif/data/lifconfig.h>

#include <modules/lif/data/liftrace.h>
#include <QFile>
#include <cmath>


LifConfig::LifConfig() : HeaderStorage(BC::Store::LIF::key)
{
    SettingsStorage s(BC::Key::hwKey(BC::Key::LifDigi::lifScope,0),SettingsStorage::Hardware);
    QString sk = s.get(BC::Key::HW::subKey,BC::Key::Comm::hwVirtual);
    ps_scopeConfig = std::make_shared<LifDigitizerConfig>(sk);
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
    return qMakePair(d_delayStartUs,d_delayStartUs + d_delayStepUs*(d_delayPoints-1));
}

QPair<double, double> LifConfig::laserRange() const
{
    return qMakePair(d_laserPosStart,d_laserPosStart + d_laserPosStep*(d_laserPosPoints-1));
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
    return {d_procSettings.lifGateStart,d_procSettings.lifGateEnd};
}

QPair<int, int> LifConfig::refGate() const
{
    return {d_procSettings.refGateStart,d_procSettings.refGateEnd};
}

void LifConfig::addWaveform(const QVector<qint8> d)
{
    //the boolean returned by this function tells if the point was incremented
    if(d_complete && d_completeMode == StopWhenComplete)
        return;

    LifTrace t(scopeConfig(),d,d_currentDelayIndex,d_currentLaserIndex);
    ps_storage->addTrace(t);
}

void LifConfig::loadLifData()
{
    ps_storage = std::make_shared<LifStorage>(d_delayPoints,d_laserPosPoints,d_number,d_path);
    LifTrace::LifProcSettings s;
    if(ps_storage->readProcessingSettings(s))
        d_procSettings = s;
    ps_storage->finish();
}

void LifConfig::storeValues()
{
    SettingsStorage s(BC::Key::LifLaser::key,SettingsStorage::Hardware);
    auto lUnits = s.get(BC::Key::LifLaser::units,QString("nm"));

    using namespace BC::Store::LIF;
    store(order,d_order);
    store(completeMode,d_completeMode);
    store(dStart,d_delayStartUs,BC::Unit::us);
    store(dStep,d_delayStepUs,BC::Unit::us);
    store(dPoints,d_delayPoints);
    store(lStart,d_laserPosStart,lUnits);
    store(lStep,d_laserPosStep,lUnits);
    store(lPoints,d_laserPosPoints);
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
    addChild(&scopeConfig());
}


bool LifConfig::initialize()
{
    ps_storage = std::make_shared<LifStorage>(d_delayPoints,d_laserPosPoints,d_number,d_path);
    ps_storage->writeProcessingSettings(d_procSettings);
    ps_storage->start();
    d_processingPaused = true;
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
