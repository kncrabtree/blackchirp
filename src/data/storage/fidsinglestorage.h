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
    FidList getFidList(std::size_t i) override;
    FidList getCurrentFidList() override;
    int getCurrentIndex() override;
    void autoSave() override;

#ifdef BC_CUDA
    bool setFidsData(const FidList other) override;
#endif
    // FidStorageBase interface
protected:
    void _advance() override;

private:
    QMutex *p_mutex;
    FidList d_currentFidList;
    int d_lastAutosave{-1};


};


#endif // FIDSINGLESTORAGE_H
