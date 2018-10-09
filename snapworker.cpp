#include "snapworker.h"

#include <QFile>
#include <QDataStream>
#include <QByteArray>

#include "datastructs.h"

SnapWorker::SnapWorker(QObject *parent) : QObject(parent)
{

}

void SnapWorker::calculateSnapshots(FtmwConfig allFids, const QList<int> snaps, bool includeRemainder, int num, QString path)
{

    if(includeRemainder)
    {
        FtmwConfig copy = allFids;
        copy.loadFidsFromSnapshots(num,path,snaps);
        allFids.subtractFids(copy);
    }
    else
        allFids.loadFidsFromSnapshots(num,path,snaps);

    emit fidsUpdated(allFids);
}

