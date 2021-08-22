#ifndef FIDPEAKUPSTORAGE_H
#define FIDPEAKUPSTORAGE_H

#include <data/storage/fidstoragebase.h>

class QMutex;

class FidPeakUpStorage : public FidStorageBase
{
public:
    FidPeakUpStorage(int numRecords);
    ~FidPeakUpStorage();

    // FidStorageBase interface
    bool addFids(const FidList other, int shift) override;
    int getCurrentIndex() override;
    bool setFidsData(const FidList other) override;

    void reset();
    void setTargetShots(quint64 s);
    quint64 targetShots() const;


private:
    quint64 d_targetShots{1};

};

#endif // FIDPEAKUPSTORAGE_H
