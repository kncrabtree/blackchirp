#include <acquisition/savemanager.h>

#include <QFile>

#include <data/datastructs.h>

SaveManager::SaveManager(QObject *parent) : QObject(parent), d_snapNum(0)
{

}

void SaveManager::snapshot(Experiment &e)
{
    (void)e;
#pragma message ("Snapshot implementation still needed")
//    e.snapshot(d_snapNum,d_lastExperiment);
//    d_snapNum++;
//    d_lastExperiment = e;

//    QFile snp(BlackChirp::getExptFile(e.number(),BlackChirp::SnapFile));
//    if(snp.open(QIODevice::WriteOnly))
//    {
//        if(e.ftmwEnabled())
//        {
//            if(e.ftmwConfig()->hasMultiFidLists())
//                snp.write(QByteArray("mfd\t") + QByteArray::number(d_snapNum) + QByteArray("\n"));
//            else
//                snp.write(QByteArray("fid\t") + QByteArray::number(d_snapNum) + QByteArray("\n"));
//        }
//        snp.close();
//    }

    emit snapshotComplete();
}

void SaveManager::finalSave(Experiment &e)
{
    //consider making finalsave return bool for error signaling?
    e.finalSave();
    emit finalSaveComplete(e);
}

