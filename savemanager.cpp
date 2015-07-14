#include "savemanager.h"

#include <QFile>

#include "datastructs.h"

SaveManager::SaveManager(QObject *parent) : QObject(parent), d_snapNum(0)
{

}

void SaveManager::snapshot(const Experiment e)
{
    e.snapshot(d_snapNum,d_lastExperiment);
    d_snapNum++;
    d_lastExperiment = e;

    QFile snp(BlackChirp::getExptFile(e.number(),BlackChirp::SnapFile));
    if(snp.open(QIODevice::WriteOnly))
    {
        snp.write(QByteArray("fid\t") + QByteArray::number(d_snapNum) + QByteArray("\n"));
        snp.close();
    }

    emit snapshotComplete();
}

void SaveManager::finalSave(const Experiment e)
{
    //consider making finalsave return bool for error signaling?
    e.finalSave();
    emit finalSaveComplete(e);
}

