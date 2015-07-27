#include "snapworker.h"

#include <QFile>
#include <QDataStream>
#include <QByteArray>

#include "datastructs.h"

SnapWorker::SnapWorker(QObject *parent) : QObject(parent)
{

}

void SnapWorker::calculateFidList(int exptNum, const QList<int> snapList, bool subtractFromFull)
{
    QList<Fid> out;

    if(subtractFromFull)
    {
        out = parseFile(exptNum,-1);
        if(out.isEmpty())
        {
            emit fidListComplete(QList<Fid>());
            return;
        }

        if(snapList.isEmpty())
        {
            emit fidListComplete(out);
            return;
        }

        for(int i=0; i<snapList.size(); i++)
        {
            QList<Fid> snap = parseFile(exptNum,snapList.at(i));
            if(snap.size() != out.size())
            {
                emit fidListComplete(QList<Fid>());
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

        out = parseFile(exptNum,snapList.first());
        if(out.isEmpty())
        {
            emit fidListComplete(out);
            return;
        }

        for(int i=1; i<snapList.size(); i++)
        {
            QList<Fid> snap = parseFile(exptNum,snapList.at(i));
            if(snap.size() != out.size())
            {
                emit fidListComplete(QList<Fid>());
                return;
            }
            for(int j=0; j<snap.size(); j++)
                out[j] += snap.at(j);
        }
    }

    emit fidListComplete(out);
}

QList<Fid> SnapWorker::parseFile(int exptNum, int snapNum, QString path)
{
    QList<Fid> out;

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

