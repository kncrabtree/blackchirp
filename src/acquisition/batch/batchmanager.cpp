#include <acquisition/batch/batchmanager.h>

BatchManager::BatchManager(BatchManager::BatchType b)
    : QObject(), d_type(b)
{

}

BatchManager::~BatchManager()
{

}

void BatchManager::experimentComplete()
{
    auto exp = currentExperiment();
    if(!exp->errorString().isEmpty())
        emit logMessage(exp->errorString(),BlackChirp::LogError);

    //as of v1.0 these conditions are not possible I think
//    if(!exp->isInitialized() || !exp->hardwareSuccess())
//    {
//        writeReport();
//        emit batchComplete(true);
//        return;
//    }

    emit logMessage(exp->endLogMessage(),exp->endLogMessageCode());

    processExperiment();
    if(!exp->isAborted() && !isComplete())
        beginNextExperiment();
    else
    {
        writeReport();
        emit batchComplete(exp->isAborted());
    }

}

void BatchManager::beginNextExperiment()
{
    emit beginExperiment();
}

