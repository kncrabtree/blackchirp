#ifndef BATCHSINGLE_H
#define BATCHSINGLE_H

#include <acquisition/batch/batchmanager.h>

class BatchSingle : public BatchManager
{
public:
    BatchSingle(std::shared_ptr<Experiment> e);
    ~BatchSingle();

public slots:
     void abort();

    // BatchManager interface
protected:
    void writeReport();
    void processExperiment();
    std::shared_ptr<Experiment> currentExperiment();
    bool isComplete();

    std::shared_ptr<Experiment> d_exp;
    bool d_complete;
};

#endif // BATCHSINGLE_H
