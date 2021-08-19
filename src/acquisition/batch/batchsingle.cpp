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
    d_complete = true;
}

void BatchSingle::writeReport()
{
    //no report to write
}

void BatchSingle::processExperiment()
{
    d_complete = true;
}

std::shared_ptr<Experiment> BatchSingle::currentExperiment()
{
    return d_exp;
}

bool BatchSingle::isComplete()
{
    return d_complete;
}
