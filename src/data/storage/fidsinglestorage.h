#ifndef FIDSINGLESTORAGE_H
#define FIDSINGLESTORAGE_H

class QMutex;

#include <data/storage/fidstoragebase.h>

class FidSingleStorage : public FidStorageBase
{
public:
    FidSingleStorage(int numRecords, int num, QString path="");

    // FidStorageBase interface
    virtual FidList loadDifferentialFidList(int i);
    int getCurrentIndex() override;
    void backup() override;
    int numBackups() override;

private:
    int d_lastBackup{0};

};


#endif // FIDSINGLESTORAGE_H
