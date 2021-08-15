#include <data/experiment/ftmwconfigtypes.h>

#include <data/storage/fidsinglestorage.h>
#include <data/storage/fidpeakupstorage.h>


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
    return std::make_shared<FidSingleStorage>(d_scopeConfig.d_numRecords,num,path);
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
    return std::make_shared<FidPeakUpStorage>(d_scopeConfig.d_numRecords);
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
    return std::make_shared<FidSingleStorage>(d_scopeConfig.d_numRecords,num,path);
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
    return std::make_shared<FidSingleStorage>(d_scopeConfig.d_numRecords,num,path);
}
