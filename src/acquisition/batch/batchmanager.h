#ifndef BATCHMANAGER_H
#define BATCHMANAGER_H

#include <QObject>

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
    virtual bool isComplete() =0;

signals:
    void statusMessage(QString,int=0);
    void logMessage(QString,LogHandler::MessageCode = LogHandler::Normal);
    void beginExperiment();
    void batchComplete(bool aborted);

public slots:
    void experimentComplete();
    virtual void beginNextExperiment();
    virtual void abort() =0;

protected:
    BatchType d_type;

    virtual void writeReport() =0;
    virtual void processExperiment() =0;
};

#endif //BATCHMANAGER_H
