#include "batchmanager.h"

BatchManager::BatchManager(BatchManager::BatchType b)
    : QObject(), d_type(b)
{

}

BatchManager::~BatchManager()
{

}

void BatchManager::experimentComplete(const Experiment exp)
{
    if(!exp.isInitialized())
    {
        writeReport();
        emit batchComplete(true);
        return;
    }

    if(!exp.isDummy())
        emit logMessage(exp.endLogMessage(),exp.endLogMessageCode());

    processExperiment(exp);
    if(!exp.isAborted() && !isComplete())
        beginNextExperiment();
    else
    {
        writeReport();
        emit batchComplete(exp.isAborted());
    }

}

void BatchManager::beginNextExperiment()
{
    emit beginExperiment(nextExperiment());
}

