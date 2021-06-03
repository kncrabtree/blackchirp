#ifndef BATCHSINGLE_H
#define BATCHSINGLE_H

#include "batchmanager.h"

class BatchSingle : public BatchManager
{
public:
    BatchSingle(const Experiment e);
    ~BatchSingle();

public slots:
     void abort();

    // BatchManager interface
protected:
    void writeReport();
    void processExperiment(const Experiment exp);
    Experiment nextExperiment();
    bool isComplete();

    Experiment d_exp;
    bool d_complete;
};

#endif // BATCHSINGLE_H
