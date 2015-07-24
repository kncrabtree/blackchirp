#include "batchmanager.h"

BatchManager::BatchManager(BatchManager::BatchType b)
    : QObject(), d_type(b), d_sleep(false)
{

}

BatchManager::~BatchManager()
{

}

void BatchManager::experimentComplete(const Experiment exp)
{
    if(!exp.errorString().isEmpty())
        emit logMessage(exp.errorString(),BlackChirp::LogError);

    if(!exp.isInitialized() || !exp.hardwareSuccess())
    {
        writeReport();
        emit batchComplete(true);
        return;
    }

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

