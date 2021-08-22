#include "fidsinglestorage.h"

#include <QMutexLocker>
#include <data/storage/blackchirpcsv.h>

FidSingleStorage::FidSingleStorage(int numRecords, int num, QString path) : FidStorageBase(numRecords,num,path)
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
                d_lastBackup++;
        }
    }
}

int FidSingleStorage::getCurrentIndex()
{
    return 0;
}

void FidSingleStorage::backup()
{
    //note: d_lastBackup starts at 0, so this increments it to 1 on first call.
    //backups will be numbered starting from 1, and final data will be numbered 0
    ++d_lastBackup;

    auto fl = getCurrentFidList();
    saveFidList(fl,d_lastBackup);
}

int FidSingleStorage::numBackups()
{
    return d_lastBackup;
}
