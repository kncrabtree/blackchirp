#ifndef FIDSTORAGEBASE_H
#define FIDSTORAGEBASE_H

#include <memory>
#include <queue>
#include <QDateTime>
#include <QMutex>

#include <data/experiment/fid.h>

class BlackchirpCSV;

class FidStorageBase
{

public:
    FidStorageBase(int numRecords, int number = -1, QString path = "");
    virtual ~FidStorageBase();

    const int d_number;
    const int d_numRecords;
    const QString d_path;


    virtual quint64 completedShots() =0;
    virtual quint64 currentSegmentShots() =0;
    virtual bool addFids(const FidList other, int shift =0) =0;
    virtual FidList getCurrentFidList() =0;
    virtual void backup() { return; };
    virtual int numBackups() { return 0; }
    void advance();
    void save();
    void start();
    void finish();
    virtual bool setFidsData(const FidList other) =0;
    virtual int getCurrentIndex() =0;
    FidList loadFidList(int i);

protected:
    virtual void _advance() =0;
    void saveFidList(const FidList l, int i);

private:
    bool d_acquiring{false};
    int d_currentSegment{0};
    std::size_t d_maxCacheSize{1 << 25}; //~200 MB
    QVector<Fid> d_templateList;
    std::unique_ptr<BlackchirpCSV> pu_csv;
    std::unique_ptr<QMutex> pu_mutex;
    std::queue<int> d_cacheKeys;
    std::map<int,FidList> d_cache;

};

#endif // FIDSTORAGEBASE_H
