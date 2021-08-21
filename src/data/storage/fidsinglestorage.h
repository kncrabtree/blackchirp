#ifndef FIDSINGLESTORAGE_H
#define FIDSINGLESTORAGE_H

#include <QMutex>

#include <data/storage/fidstoragebase.h>

class FidSingleStorage : public FidStorageBase
{
public:
    FidSingleStorage(int numRecords, int num, QString path="");
    ~FidSingleStorage();

    // FidStorageBase interface
    quint64 completedShots() override;
    quint64 currentSegmentShots() override;
    bool addFids(const FidList other, int shift) override;
    FidList getCurrentFidList() override;
    int getCurrentIndex() override;
    void backup() override;
    int numBackups() override;
    bool setFidsData(const FidList other) override;

    // FidStorageBase interface
protected:
    void _advance() override;

private:
    QMutex *p_mutex;
    FidList d_currentFidList;
    int d_lastBackup{0};

};


#endif // FIDSINGLESTORAGE_H
