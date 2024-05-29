#include <acquisition/batch/batchsequence.h>

#include <QDir>

BatchSequence::BatchSequence(std::shared_ptr<Experiment> e, int numExpts, int intervalSeconds) :
    BatchManager(BatchManager::Sequence), d_numExperiments(numExpts),
    d_intervalSeconds(intervalSeconds), d_waiting(false)
{
    pu_expTemplate = std::make_unique<Experiment>(*e.get());
    ps_CurrentExp = e;

    p_intervalTimer = new QTimer(this);
    p_intervalTimer->setSingleShot(true);
    connect(p_intervalTimer,&QTimer::timeout,this,[this](){
        d_waiting = false;
        ps_CurrentExp = std::make_shared<Experiment>(*pu_expTemplate.get());
        emit beginExperiment();
    });
}



void BatchSequence::abort()
{
    if(d_waiting)
    {
        d_experimentCount = d_numExperiments;
        p_intervalTimer->stop();
        p_intervalTimer->blockSignals(true);
        emit batchComplete(true);
        return;
    }
}

void BatchSequence::writeReport()
{
    //not generating reports for this
}

void BatchSequence::processExperiment()
{
    d_experimentCount++;
}

std::shared_ptr<Experiment> BatchSequence::currentExperiment()
{
    return ps_CurrentExp;
}

bool BatchSequence::isComplete()
{
    return d_experimentCount >= d_numExperiments;
}

void BatchSequence::beginNextExperiment()
{
    //start a timer, and emit the beginExperiment signal when done
    //set waiting to true so that this can be aborted if necessary
    if(d_experimentCount == 0)
    {
        emit beginExperiment();
    }
    else if(d_experimentCount < d_numExperiments)
    {
        d_waiting = true;
        p_intervalTimer->start(d_intervalSeconds*1000);
        emit statusMessage(QString("Next experiment will start at %1").arg(QDateTime::currentDateTime().addSecs(d_intervalSeconds).toString()));
    }
}
