#ifndef FIDSTORAGEBASE_H
#define FIDSTORAGEBASE_H

#include <queue>
#include <QDateTime>

#include <data/storage/datastoragebase.h>
#include <data/experiment/fid.h>

class BlackchirpCSV;

class FidStorageBase : public DataStorageBase
{

public:
    FidStorageBase(int numRecords, int number = -1, QString path = "");
    virtual ~FidStorageBase();

    const int d_numRecords;

    void advance() override;
    void save() override;
    void start() override;
    void finish() override;
    FidList loadFidList(int i);

    virtual quint64 currentSegmentShots();
    virtual bool addFids(const FidList other, int shift =0);
    virtual bool setFidsData(const FidList other);
    virtual FidList getCurrentFidList();
    virtual void backup() { return; };
    virtual int numBackups() { return 0; }
    virtual int getCurrentIndex() =0;

protected:
    FidList d_currentFidList;
    virtual void _advance() {};
    void saveFidList(const FidList l, int i);

private:
    bool d_acquiring{false};
    int d_currentSegment{0};
    std::size_t d_maxCacheSize{1 << 25}; //~200 MB
    QVector<Fid> d_templateList;
    std::unique_ptr<QMutex> pu_baseMutex;
    std::queue<int> d_cacheKeys;
    std::map<int,FidList> d_cache;

    void updateCache(const FidList fl, int i);

};

#endif // FIDSTORAGEBASE_H
