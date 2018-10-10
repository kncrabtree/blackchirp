#include "snapworker.h"

#include <QFile>
#include <QDataStream>
#include <QByteArray>

#include "datastructs.h"

SnapWorker::SnapWorker(QObject *parent) : QObject(parent)
{

}

FtmwConfig SnapWorker::calculateSnapshots(FtmwConfig allFids, const QList<int> snaps, bool includeRemainder, int num, QString path)
{

    if(includeRemainder)
    {
        FtmwConfig copy = allFids;
        copy.loadFidsFromSnapshots(num,path,snaps);
        allFids.subtractFids(copy);
    }
    else
        allFids.loadFidsFromSnapshots(num,path,snaps);

    emit processingComplete(allFids);
    return allFids;
}

void SnapWorker::finalizeSnapshots(FtmwConfig allFids, const QList<int> snaps, bool includeRemainder, int num, QString path)
{
    blockSignals(true);
    FtmwConfig out = calculateSnapshots(allFids,snaps,includeRemainder,num,path);
    blockSignals(false);

    emit finalProcessingComplete(out);
}

