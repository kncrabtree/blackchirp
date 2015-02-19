#ifndef BATCHSINGLE_H
#define BATCHSINGLE_H

#include "batchmanager.h"

class BatchSingle : public BatchManager
{
public:
    BatchSingle(Experiment e);
    ~BatchSingle();

    // BatchManager interface
protected:
    void writeReport();
    void processExperiment(Experiment exp);
    Experiment nextExperiment();
    bool isComplete();

    Experiment d_exp;
    bool d_complete;
};

#endif // BATCHSINGLE_H
