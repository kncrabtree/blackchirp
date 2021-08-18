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
    if(!exp->d_errorString.isEmpty())
        emit logMessage(exp->d_errorString,LogHandler::Error);

    //as of v1.0 these conditions are not possible I think
//    if(!exp->isInitialized() || !exp->hardwareSuccess())
//    {
//        writeReport();
//        emit batchComplete(true);
//        return;
//    }

    emit logMessage(exp->d_endLogMessage,exp->d_endLogMessageCode);

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

