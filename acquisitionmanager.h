#ifndef ACQUISITIONMANAGER_H
#define ACQUISITIONMANAGER_H

#include <QObject>
#include "loghandler.h"
#include "experiment.h"

class AcquisitionManager : public QObject
{
    Q_OBJECT
public:
    explicit AcquisitionManager(QObject *parent = 0);
    ~AcquisitionManager();

    enum AcquisitionState
    {
        Idle,
        Acquiring,
        Paused
    };

signals:
    void logMessage(const QString,const LogHandler::MessageCode = LogHandler::Normal);
    void statusMessage(const QString);
    void experimentComplete(Experiment);
    void ftmwShotAcquired(int);
    void beginAcquisition();

public slots:
    void startExperiment(Experiment exp);
    void processScopeShot(const QByteArray b);

private:
    Experiment d_currentExperiment;
    AcquisitionState d_state;

    void checkComplete();
};

#endif // ACQUISITIONMANAGER_H
