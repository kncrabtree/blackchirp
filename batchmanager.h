#ifndef BATCHMANAGER_H
#define BATCHMANAGER_H

#include <QObject>

#include "loghandler.h"
#include "experiment.h"

class BatchManager : public QObject
{
    Q_OBJECT
public:
    enum BatchType
    {
        SingleExperiment
    };

    explicit BatchManager(BatchType b);
    ~BatchManager();

signals:
    void logMessage(QString,LogHandler::MessageCode = LogHandler::Normal);
    void beginExperiment(Experiment);
    void batchComplete(bool aborted);

public slots:
    void experimentComplete(Experiment exp);
    void beginNextExperiment();

protected:
    BatchType d_type;

    virtual void writeReport() =0;
    virtual void processExperiment(Experiment exp) =0;
    virtual Experiment nextExperiment() =0;
    virtual bool isComplete() =0;
};

#endif //BATCHMANAGER_H
