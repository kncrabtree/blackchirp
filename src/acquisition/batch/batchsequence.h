#ifndef BATCHSEQUENCE_H
#define BATCHSEQUENCE_H

#include "batchmanager.h"

#include <QTimer>

class BatchSequence : public BatchManager
{
public:
    BatchSequence();

    void setExperiment(const Experiment e) { d_exp = e; }
    void setNumExperiments(int num) { d_numExperiments = num; }
    void setInterval(int seconds) { d_intervalSeconds = seconds; }
    void setAutoExport(bool exp) { d_autoExport = exp; }
    void setExportPath(QString path) { d_exportPath = path; }

private:
    Experiment d_exp;
    int d_experimentCount;
    int d_numExperiments;
    int d_intervalSeconds;
    bool d_autoExport;
    QString d_exportPath;
    bool d_waiting;
    QTimer *p_intervalTimer;

    // BatchManager interface
public slots:
    void abort();
    void beginNextExperiment();

protected:
    void writeReport();
    void processExperiment(const Experiment exp);
    Experiment nextExperiment();
    bool isComplete();

};

#endif // BATCHSEQUENCE_H
