#ifndef FIDSTORAGEBASE_H
#define FIDSTORAGEBASE_H

#include <queue>
#include <QDateTime>

#include <data/storage/datastoragebase.h>
#include <data/experiment/fid.h>
#include <data/analysis/ftworker.h>

class BlackchirpCSV;

namespace BC::Key::FidStorage {
static const QString fidStart{"FidStartUs"};
static const QString fidEnd{"FidEndUs"};
static const QString fidExp{"FidExpfUs"};
static const QString zpf{"FidZeroPadFactor"};
static const QString rdc{"FidRemoveDC"};
static const QString units{"FtUnits"};
static const QString autoscaleIgnore{"AutoscaleIgnoreMHz"};
static const QString winf{"FidWindowFunction"};
}

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
    virtual void backup() { return; }
    virtual int numBackups() { return 0; }
    virtual int getCurrentIndex() =0;

    void writeProcessingSettings(const FtWorker::FidProcessingSettings &c);
    bool readProcessingSettings(FtWorker::FidProcessingSettings &out);
    std::pair<double,double> getLORange();

protected:
    FidList d_currentFidList;
    virtual void _advance() {}
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

