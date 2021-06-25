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

    explicit BatchManager(BatchType b);
    virtual ~BatchManager();

signals:
    void statusMessage(QString);
    void logMessage(QString,BlackChirp::LogMessageCode = BlackChirp::LogNormal);
    void beginExperiment(Experiment);
    void batchComplete(bool aborted);

public slots:
    void experimentComplete(const Experiment exp);
    virtual void beginNextExperiment();

    //NOTE: Abort is only used if user wants to stop _between_ experiments.
    //If abort happens during an experiment, the experimentComplete() function will handle it
    virtual void abort() =0;

protected:
    BatchType d_type;

    virtual void writeReport() =0;
    virtual void processExperiment(const Experiment exp) =0;
    virtual Experiment nextExperiment() =0;
    virtual bool isComplete() =0;
};

#endif //BATCHMANAGER_H
