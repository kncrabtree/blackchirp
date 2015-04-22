#include "batchsingle.h"

BatchSingle::BatchSingle(const Experiment e) : BatchManager(BatchManager::SingleExperiment), d_exp(e), d_complete(false)
{
}

BatchSingle::~BatchSingle()
{

}



void BatchSingle::writeReport()
{
    //no report to write
}

void BatchSingle::processExperiment(const Experiment exp)
{
    //no processing needs to be done
    Q_UNUSED(exp)
}

Experiment BatchSingle::nextExperiment()
{
    //only one experiment, so set complete to true and return it
    d_complete = true;
    return d_exp;

}

bool BatchSingle::isComplete()
{
    return d_complete;
}
