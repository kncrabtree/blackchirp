#ifndef BATCHMANAGER_H
#define BATCHMANAGER_H

#include <QObject>

#include "datastructs.h"
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
    virtual ~BatchManager();

    void setSleep(bool s) { d_sleep = s; }
    bool sleepWhenComplete() const { return d_sleep; }

signals:
    void logMessage(QString,BlackChirp::LogMessageCode = BlackChirp::LogNormal);
    void beginExperiment(Experiment);
    void batchComplete(bool aborted);

public slots:
    void experimentComplete(const Experiment exp);
    void beginNextExperiment();

protected:
    BatchType d_type;
    bool d_sleep;

    virtual void writeReport() =0;
    virtual void processExperiment(const Experiment exp) =0;
    virtual Experiment nextExperiment() =0;
    virtual bool isComplete() =0;
};

#endif //BATCHMANAGER_H
