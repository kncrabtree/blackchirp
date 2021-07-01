#ifndef FIDSTORAGEBASE_H
#define FIDSTORAGEBASE_H

#include <deque>
#include <map>

#include <data/experiment/fid.h>

class FidStorageBase
{

public:
    FidStorageBase(const QString path, int numRecords);
    virtual ~FidStorageBase();

    const QString d_path;
    const int d_numRecords;

    virtual quint64 completedShots() =0;
    virtual quint64 currentSegmentShots() =0;
    virtual bool addFids(const FidList other, int shift =0) =0;
    virtual FidList getFidList(std::size_t i=0) =0;
    virtual FidList getCurrentFidList() =0;
    virtual void reset() =0;
    void advance();
    bool save();
#ifdef BC_CUDA
    virtual bool setFidsData(const FidList other) =0;
#endif
    virtual int getCurrentIndex() =0;

protected:
    virtual void _advance() =0;

    FidList newFidList() const;

private:
    bool saveFidList(const FidList l, int i);

};
//std::deque<std::map<int,FidList>::iterator> d_cache;
//std::map<int,FidList> d_memStorage;
//std::vector<quint64> d_completedShots;
//int d_currentIndex;

#endif // FIDSTORAGEBASE_H
