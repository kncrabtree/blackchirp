#ifndef FIDPEAKUPSTORAGE_H
#define FIDPEAKUPSTORAGE_H

#include <data/storage/fidstoragebase.h>
#include <QMutex>


class FidPeakUpStorage : public FidStorageBase
{
public:
    FidPeakUpStorage(int numRecords);
    ~FidPeakUpStorage();

    // FidStorageBase interface
    quint64 completedShots() override;
    quint64 currentSegmentShots() override;
    bool addFids(const FidList other, int shift) override;
    FidList getFidList(std::size_t i) override;
    FidList getCurrentFidList() override;
    int getCurrentIndex() override;
#ifdef BC_CUDA
    bool setFidsData(const FidList other) override;
#endif

    void reset();
    void setTargetShots(quint64 s);

protected:
    void _advance() override;

private:
    quint64 d_targetShots{1};
    FidList d_currentFidList;

    QMutex *p_mutex;

    // FidStorageBase interface
public:

protected:
};

#endif // FIDPEAKUPSTORAGE_H
