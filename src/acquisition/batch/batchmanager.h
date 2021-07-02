#ifndef BATCHMANAGER_H
#define BATCHMANAGER_H

#include <QObject>

#include <data/datastructs.h>
#include <data/experiment/experiment.h>

class BatchManager : public QObject
{
    Q_OBJECT
public:
    enum BatchType
    {
        SingleExperiment,
        Sequence
    };

    virtual std::shared_ptr<Experiment> currentExperiment() =0;
    explicit BatchManager(BatchType b);
    virtual ~BatchManager();

signals:
    void statusMessage(QString);
    void logMessage(QString,BlackChirp::LogMessageCode = BlackChirp::LogNormal);
    void beginExperiment();
    void batchComplete(bool aborted);

public slots:
    void experimentComplete();
    virtual void beginNextExperiment();

    //NOTE: Abort is only used if user wants to stop _between_ experiments.
    //If abort happens during an experiment, the experimentComplete() function will handle it
    virtual void abort() =0;

protected:
    BatchType d_type;

    virtual void writeReport() =0;
    virtual void processExperiment() =0;
    virtual bool isComplete() =0;
};

#endif //BATCHMANAGER_H
