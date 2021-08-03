#ifndef FIDSTORAGEBASE_H
#define FIDSTORAGEBASE_H

#include <QFuture>
#include <memory>

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
    virtual FidList getFidList(std::size_t i=0) =0;
    virtual FidList getCurrentFidList() =0;
    virtual void autoSave() =0;
    void advance();
    QFuture<void> save();
#ifdef BC_CUDA
    virtual bool setFidsData(const FidList other) =0;
#endif
    virtual int getCurrentIndex() =0;

protected:
    virtual void _advance() =0;
    QFuture<void> saveFidList(const FidList l, int i);
    QFuture<FidList> loadFidList(int i);

private:
    QVector<Fid> d_templateList;
    std::unique_ptr<BlackchirpCSV> pu_csv;

};

#endif // FIDSTORAGEBASE_H
