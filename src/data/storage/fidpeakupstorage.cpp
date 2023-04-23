#include "fidpeakupstorage.h"

FidPeakUpStorage::FidPeakUpStorage(int numRecords) :
    FidStorageBase(numRecords)
{
    pu_mutex = std::make_unique<QMutex>();
}

FidPeakUpStorage::~FidPeakUpStorage()
{
}

bool FidPeakUpStorage::addFids(const FidList other, int shift)
{
    QMutexLocker l(pu_mutex.get());
    if(d_currentFidList.isEmpty())
        d_currentFidList = other;
    else
    {
        if(other.size() != d_currentFidList.size())
            return false;
        for(int i=0; i<d_currentFidList.size(); i++)
            d_currentFidList[i].rollingAverage(other.at(i),d_targetShots,shift);
    }

    return true;
}

void FidPeakUpStorage::reset()
{
    QMutexLocker l(pu_mutex.get());
    d_currentFidList.clear();
}

int FidPeakUpStorage::getCurrentIndex()
{
    return 0;
}

void FidPeakUpStorage::setTargetShots(quint64 s)
{
    QMutexLocker l(pu_mutex.get());
    d_targetShots = s;
}

quint64 FidPeakUpStorage::targetShots() const
{
    QMutexLocker l(pu_mutex.get());
    return d_targetShots;
}

bool FidPeakUpStorage::setFidsData(const FidList other)
{
    QMutexLocker l(pu_mutex.get());
    if(!d_currentFidList.isEmpty() && (other.size() != d_currentFidList.size()))
        return false;

    d_currentFidList = other;
    return true;
}
