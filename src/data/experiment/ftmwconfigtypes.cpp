#include <data/experiment/ftmwconfigtypes.h>

#include <data/storage/fidsinglestorage.h>
#include <data/storage/fidpeakupstorage.h>
#include <data/storage/fidmultistorage.h>


/******************************************
 *
 * FtmwConfigSingle
 *
 * ****************************************/

FtmwConfigSingle::FtmwConfigSingle() : FtmwConfig()
{
}

FtmwConfigSingle::FtmwConfigSingle(const FtmwConfig &other) : FtmwConfig(other)
{
}

int FtmwConfigSingle::perMilComplete() const
{
    return (1000*completedShots())/d_objective;
}

bool FtmwConfigSingle::isComplete() const
{
    return completedShots() >= d_objective;
}

quint64 FtmwConfigSingle::completedShots() const
{
    return storage()->currentSegmentShots();
}

bool FtmwConfigSingle::_init()
{
    return true;
}

void FtmwConfigSingle::_prepareToSave()
{
    store(BC::Store::FTMW::tShots,d_objective);
}

void FtmwConfigSingle::_loadComplete()
{
}

std::shared_ptr<FidStorageBase> FtmwConfigSingle::createStorage(int num, QString path)
{
    return std::make_shared<FidSingleStorage>(scopeConfig().d_numRecords,num,path);
}

/******************************************
 *
 * FtmwConfigPeakUp
 *
 * ****************************************/


FtmwConfigPeakUp::FtmwConfigPeakUp() : FtmwConfig()
{

}

FtmwConfigPeakUp::FtmwConfigPeakUp(const FtmwConfig &other) : FtmwConfig(other)
{
}

int FtmwConfigPeakUp::perMilComplete() const
{
    return (1000*completedShots())/d_objective;
}

bool FtmwConfigPeakUp::isComplete() const
{
    return false;
}

quint64 FtmwConfigPeakUp::completedShots() const
{
    return storage()->currentSegmentShots();
}

quint8 FtmwConfigPeakUp::bitShift() const
{
    return 8;
}

bool FtmwConfigPeakUp::_init()
{
    static_cast<FidPeakUpStorage*>(storage().get())->setTargetShots(d_objective);
    return true;
}

void FtmwConfigPeakUp::_prepareToSave()
{
    store(BC::Store::FTMW::tShots,d_objective);
}

void FtmwConfigPeakUp::_loadComplete()
{
}

std::shared_ptr<FidStorageBase> FtmwConfigPeakUp::createStorage(int num, QString path)
{
    Q_UNUSED(num)
    Q_UNUSED(path)
    return std::make_shared<FidPeakUpStorage>(scopeConfig().d_numRecords);
}


/******************************************
 *
 * FtmwConfigDuration
 *
 * ****************************************/


FtmwConfigDuration::FtmwConfigDuration() : FtmwConfig()
{

}

FtmwConfigDuration::FtmwConfigDuration(const FtmwConfig &other) : FtmwConfig(other)
{
    auto o = dynamic_cast<const FtmwConfigDuration*>(&other);
    if(o)
    {
        d_startTime = o->d_startTime;
        d_targetTime = o->d_targetTime;
    }
}

int FtmwConfigDuration::perMilComplete() const
{
    return qBound(0,
                  1000-static_cast<int>(((1000*QDateTime::currentDateTime().secsTo(d_targetTime))
                        /d_startTime.secsTo(d_targetTime) )),
                  1000);
}

bool FtmwConfigDuration::isComplete() const
{
    return QDateTime::currentDateTime() >= d_targetTime;
}

quint64 FtmwConfigDuration::completedShots() const
{
    return storage()->currentSegmentShots();
}

bool FtmwConfigDuration::_init()
{
    d_startTime = QDateTime::currentDateTime();
    d_targetTime = d_startTime.addSecs(d_objective*60);
    return true;
}

void FtmwConfigDuration::_prepareToSave()
{
    store(BC::Store::FTMW::duration,d_objective,"min");
}

void FtmwConfigDuration::_loadComplete()
{
    d_objective = retrieve(BC::Store::FTMW::duration,0);
}

std::shared_ptr<FidStorageBase> FtmwConfigDuration::createStorage(int num, QString path)
{
    return std::make_shared<FidSingleStorage>(scopeConfig().d_numRecords,num,path);
}


FtmwConfigForever::FtmwConfigForever() : FtmwConfig()
{

}

FtmwConfigForever::FtmwConfigForever(const FtmwConfig &other) : FtmwConfig(other)
{
}

int FtmwConfigForever::perMilComplete() const
{
    return 0;
}

bool FtmwConfigForever::indefinite() const
{
    return true;
}

bool FtmwConfigForever::isComplete() const
{
    return false;
}

quint64 FtmwConfigForever::completedShots() const
{
    return storage()->currentSegmentShots();
}

bool FtmwConfigForever::_init()
{
    return true;
}

void FtmwConfigForever::_prepareToSave()
{
}

void FtmwConfigForever::_loadComplete()
{
}

std::shared_ptr<FidStorageBase> FtmwConfigForever::createStorage(int num, QString path)
{
    return std::make_shared<FidSingleStorage>(scopeConfig().d_numRecords,num,path);
}

FtmwConfigLOScan::FtmwConfigLOScan() : FtmwConfig()
{
}

FtmwConfigLOScan::FtmwConfigLOScan(const FtmwConfig &other) : FtmwConfig(other)
{
    auto o = dynamic_cast<const FtmwConfigLOScan*>(&other);
    if(o)
    {
        d_upStart = o->d_upStart;
        d_upEnd = o->d_upEnd;
        d_upMaj = o->d_upMaj;
        d_upMin = o->d_upMin;
        d_downStart = o->d_downStart;
        d_downEnd = o->d_downEnd;
        d_downMaj = o->d_downMaj;
        d_downMin = o->d_downMin;
    }
}

int FtmwConfigLOScan::perMilComplete() const
{
    return (1000*completedShots())/d_rfConfig.totalShots();
}

bool FtmwConfigLOScan::isComplete() const
{
    return d_rfConfig.d_completedSweeps >= d_rfConfig.d_targetSweeps;
}

quint64 FtmwConfigLOScan::completedShots() const
{
    auto css = storage()->currentSegmentShots();
    quint64 overcount = qBound(0ull,
                               static_cast<quint64>(d_rfConfig.d_completedSweeps*d_rfConfig.d_shotsPerClockConfig),
                               css);

    return storage()->currentSegmentShots() + d_rfConfig.completedSegmentShots() - overcount;
}

bool FtmwConfigLOScan::_init()
{
    return true;
}

void FtmwConfigLOScan::_prepareToSave()
{
    using namespace BC::Store::FtmwLO;
    store(upStart,d_upStart,BC::Unit::MHz);
    store(upEnd,d_upEnd,BC::Unit::MHz);
    store(upMaj,d_upMaj);
    store(upMin,d_upMin);
    store(downStart,d_downStart,BC::Unit::MHz);
    store(downEnd,d_downEnd,BC::Unit::MHz);
    store(downMaj,d_downMaj);
    store(downMin,d_downMin);
}

void FtmwConfigLOScan::_loadComplete()
{
    using namespace BC::Store::FtmwLO;
    d_upStart = retrieve(upStart,0.0);
    d_upEnd = retrieve(upEnd,0.0);
    d_upMaj = retrieve(upMaj,0);
    d_upMin = retrieve(upMin,0);
    d_downStart = retrieve(downStart,0.0);
    d_downEnd = retrieve(downEnd,0.0);
    d_downMaj = retrieve(downMaj,0);
    d_downMin = retrieve(downMin,0);
}

std::shared_ptr<FidStorageBase> FtmwConfigLOScan::createStorage(int num, QString path)
{
    auto out = std::make_shared<FidMultiStorage>(scopeConfig().d_numRecords,num,path);
    out->setNumSegments(d_rfConfig.numSegments());
    return out;
}

FtmwConfigDRScan::FtmwConfigDRScan() : FtmwConfig()
{
}

FtmwConfigDRScan::FtmwConfigDRScan(const FtmwConfig &other)
{
    auto o = dynamic_cast<const FtmwConfigDRScan*>(&other);
    if(o)
    {
        d_start = o->d_start;
        d_step = o->d_step;
        d_numSteps = o->d_numSteps;
    }
}


int FtmwConfigDRScan::perMilComplete() const
{
    return (1000*completedShots())/d_rfConfig.totalShots();
}

bool FtmwConfigDRScan::isComplete() const
{
    return d_rfConfig.d_completedSweeps > 0;
}

quint64 FtmwConfigDRScan::completedShots() const
{
    return d_rfConfig.completedSegmentShots() + storage()->currentSegmentShots();
}

bool FtmwConfigDRScan::_init()
{
    return true;
}

void FtmwConfigDRScan::_prepareToSave()
{
    using namespace BC::Store::FtmwDR;
    store(drStart,d_start,BC::Unit::MHz);
    store(drStep,d_step,BC::Unit::MHz);
    store(drNumSteps,d_numSteps);
    store(drEnd,d_start + (d_numSteps-1)*d_step,BC::Unit::MHz);
}

void FtmwConfigDRScan::_loadComplete()
{
    using namespace BC::Store::FtmwDR;
    d_start = retrieve(drStart,0.0);
    d_step = retrieve(drStep,1.0);
    d_numSteps = retrieve(drNumSteps,2);
    //drEnd not needed; it's only there to display to the user
}

std::shared_ptr<FidStorageBase> FtmwConfigDRScan::createStorage(int num, QString path)
{
    auto out = std::make_shared<FidMultiStorage>(scopeConfig().d_numRecords,num,path);
    out->setNumSegments(d_rfConfig.numSegments());
    return out;
}
