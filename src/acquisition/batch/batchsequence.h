#ifndef BATCHSEQUENCE_H
#define BATCHSEQUENCE_H

#include <acquisition/batch/batchmanager.h>

#include <QTimer>

class BatchSequence : public BatchManager
{
public:
    BatchSequence(std::shared_ptr<Experiment> e, int numExpts, int intervalSeconds);


private:
    std::unique_ptr<Experiment> pu_expTemplate;
    std::shared_ptr<Experiment> ps_CurrentExp;
    int d_experimentCount{0};
    int d_numExperiments;
    int d_intervalSeconds;

    bool d_waiting;
    QTimer *p_intervalTimer;

    // BatchManager interface
public slots:
    void abort() override;
    void beginNextExperiment() override;

protected:
    void writeReport() override;
    void processExperiment() override;
    std::shared_ptr<Experiment> currentExperiment() override;
    bool isComplete() override;

};

#endif // BATCHSEQUENCE_H
