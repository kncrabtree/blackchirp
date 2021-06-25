#ifndef SAVEMANAGER_H
#define SAVEMANAGER_H

#include <QObject>

#include <data/experiment/experiment.h>

class SaveManager : public QObject
{
    Q_OBJECT
public:
    explicit SaveManager(QObject *parent = 0);

signals:
    void snapshotComplete();
    void finalSaveComplete(Experiment);
    void fatalSaveError(QString);

public slots:
    void snapshot(const Experiment e);
    void finalSave(const Experiment e);

private:
    int d_snapNum;
    Experiment d_lastExperiment;
};

#endif // SAVEMANAGER_H
