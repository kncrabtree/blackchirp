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


    emit logMessage(exp->d_endLogMessage,exp->d_endLogMessageCode);

    ///TODO: Break this up and make processExperiment run in another thread
    /// For now, though, no batch does any processing, so save for later
    ///
    bool initSuccess = exp->d_hardwareSuccess && exp->d_initSuccess;

    if(initSuccess)
        processExperiment();

    if(!exp->isAborted() && !isComplete() && initSuccess)
        beginNextExperiment();
    else
    {
        if(exp->isAborted() || !initSuccess)
            abort();

        ///TODO: Run in another thread.
        writeReport();
        emit batchComplete(exp->isAborted());
    }

}

void BatchManager::beginNextExperiment()
{
    emit beginExperiment();
}

