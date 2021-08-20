#ifndef FIDSTORAGEBASE_H
#define FIDSTORAGEBASE_H

#include <memory>
#include <QDateTime>

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
#ifdef BC_CUDA
    virtual bool setFidsData(const FidList other) =0;
#endif
    virtual int getCurrentIndex() =0;
    FidList loadFidList(int i);

protected:
    virtual void _advance() =0;
    void saveFidList(const FidList l, int i);

private:
    int d_currentSegment{0};
    QVector<Fid> d_templateList;
    std::unique_ptr<BlackchirpCSV> pu_csv;

};

#endif // FIDSTORAGEBASE_H
