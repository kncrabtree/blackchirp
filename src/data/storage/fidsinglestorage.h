#ifndef FIDSINGLESTORAGE_H
#define FIDSINGLESTORAGE_H

#include <QMutex>

#include <data/storage/fidstoragebase.h>

class FidSingleStorage : public FidStorageBase
{
public:
    FidSingleStorage(const QString path, int numRecords=1);
    ~FidSingleStorage();

    // FidStorageBase interface
protected:
    quint64 completedShots() override;
    quint64 currentSegmentShots() override;
    bool addFids(const FidList other, int shift) override;
    FidList getFidList(std::size_t i) override;
    FidList getCurrentFidList() override;
    void reset() override;
    void _advance() override;
    int getCurrentIndex() override;

#ifdef BC_CUDA
    bool setFidsData(const FidList other) override;
#endif

private:
    QMutex *p_mutex;
    FidList d_currentFidList;

    // FidStorageBase interface
protected:
};


#endif // FIDSINGLESTORAGE_H
