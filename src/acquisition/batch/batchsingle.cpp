#include <acquisition/batch/batchsingle.h>

BatchSingle::BatchSingle(std::shared_ptr<Experiment> e) : BatchManager(BatchManager::SingleExperiment),
    d_exp(e), d_complete(false)
{
}

BatchSingle::~BatchSingle()
{

}

void BatchSingle::abort()
{
    //nothing to do (experiment will signal that it was aborted)
}

void BatchSingle::writeReport()
{
    //no report to write
}

void BatchSingle::processExperiment()
{
}

std::shared_ptr<Experiment> BatchSingle::currentExperiment()
{
    //only one experiment, so set complete to true and return it
    d_complete = true;
    return d_exp;

}

bool BatchSingle::isComplete()
{
    return d_complete;
}
