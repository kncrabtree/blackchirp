#include "fidmultistorage.h"

#include <QMutexLocker>
#include <data/storage/blackchirpcsv.h>

FidMultiStorage::FidMultiStorage(int numRecords, int num, QString path) :
    FidStorageBase(numRecords,num,path)
{
    pu_mutex = std::make_unique<QMutex>();
    QDir d{BlackchirpCSV::exptDir(d_number,d_path)};
    if(d.exists(BC::CSV::fidDir))
    {
        d.cd(BC::CSV::fidDir);
        auto entries = d.entryList({"*.csv"});
        for(auto entry : entries)
        {
            auto num = entry.split('.').first().toInt();
            if(num>0)
                d_numSegments++;
        }
    }
}

int FidMultiStorage::getCurrentIndex()
{
    QMutexLocker l(pu_mutex.get());
    return d_currentSegment;
}

void FidMultiStorage::_advance()
{
    pu_mutex->lock();
    auto nextSegment = (d_currentSegment + 1) % d_numSegments;
    pu_mutex->unlock();
    auto fl = loadFidList(nextSegment);

    QMutexLocker l(pu_mutex.get());
    d_currentSegment = nextSegment;
    d_currentFidList = fl;
}
