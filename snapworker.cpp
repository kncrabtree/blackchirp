#include "snapworker.h"

#include <QFile>
#include <QDataStream>
#include <QByteArray>

#include "datastructs.h"

SnapWorker::SnapWorker(QObject *parent) : QObject(parent)
{

}

void SnapWorker::calculateFidList(int exptNum, const FidList allList, const QList<int> snapList, bool subtractFromFull)
{
    FidList out = allList;

    if(subtractFromFull)
    {
        if(snapList.isEmpty())
        {
            emit fidListComplete(out);
            return;
        }

        for(int i=0; i<snapList.size(); i++)
        {
            FidList snap = parseFile(exptNum,snapList.at(i));
            if(snap.size() != out.size())
            {
                emit fidListComplete(out);
                return;
            }
            for(int j=0; j<snap.size(); j++)
                out[j] -= snap.at(j);
        }
    }
    else
    {
        if(snapList.isEmpty())
        {
            emit fidListComplete(out);
            return;
        }

        out = parseFile(exptNum,snapList.constFirst());
        if(out.isEmpty())
        {
            emit fidListComplete(out);
            return;
        }

        for(int i=1; i<snapList.size(); i++)
        {
            FidList snap = parseFile(exptNum,snapList.at(i));
            if(snap.size() != out.size())
            {
                emit fidListComplete(allList);
                return;
            }
            for(int j=0; j<snap.size(); j++)
                out[j] += snap.at(j);
        }
    }

    emit fidListComplete(out);
}

FidList SnapWorker::parseFile(int exptNum, int snapNum, QString path)
{
    FidList out;

    QFile f(BlackChirp::getExptFile(exptNum,BlackChirp::FidFile,path,snapNum));
    if(f.open(QIODevice::ReadOnly))
    {
        QDataStream d(&f);
        QByteArray magic;
        d >> magic;
        if(magic.startsWith("BCFID"))
        {
            if(magic.endsWith("v1.0"))
                d >> out;
        }
        f.close();
    }

    return out;
}

