#include "fidsinglestorage.h"

FidSingleStorage::FidSingleStorage(int numRecords, int num, QString path) : FidStorageBase(numRecords,num,path),
    p_mutex(new QMutex)
{

    ///TODO: attempt to load from disk!
}

FidSingleStorage::~FidSingleStorage()
{
    delete p_mutex;
}

quint64 FidSingleStorage::completedShots()
{
    QMutexLocker l(p_mutex);
    if(d_currentFidList.isEmpty())
        return 0;

    return d_currentFidList.constFirst().shots();
}

quint64 FidSingleStorage::currentSegmentShots()
{
    return completedShots();
}

bool FidSingleStorage::addFids(const FidList other, int shift)
{
    QMutexLocker l(p_mutex);
    if(d_currentFidList.isEmpty())
        d_currentFidList = other;
    else
    {
        if(other.size() != d_currentFidList.size())
            return false;
        for(int i=0; i<d_currentFidList.size(); i++)
            d_currentFidList[i].add(other.at(i),shift);
    }

    return true;
}

#ifdef BC_CUDA
bool FidSingleStorage::setFidsData(const FidList other)
{
    QMutexLocker l(p_mutex);
    if(!d_currentFidList.isEmpty() && (other.size() != d_currentFidList.size()))
        return false;

    d_currentFidList = other;
    return true;
}
#endif

void FidSingleStorage::_advance()
{
}

FidList FidSingleStorage::getFidList(std::size_t i)
{
    Q_UNUSED(i)
    QMutexLocker l(p_mutex);
    return d_currentFidList;
}

FidList FidSingleStorage::getCurrentFidList()
{
    QMutexLocker l(p_mutex);
    return d_currentFidList;
}

int FidSingleStorage::getCurrentIndex()
{
    return 0;
}

void FidSingleStorage::autoSave()
{
    //note: d_lastAutosave starts at 0, so this increments it to 1 on first call.
    //autosaves will be numbered starting from 1, and final data will be numbered 0
    ++d_lastAutosave;

    auto fl = getCurrentFidList();
    saveFidList(fl,d_lastAutosave);
}
