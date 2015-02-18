#ifndef EXPERIMENT_H
#define EXPERIMENT_H

#include <QSharedDataPointer>
#include <QPair>
#include <QDateTime>

class ExperimentData;

class Experiment
{
public:
    Experiment();
    Experiment(const Experiment &);
    Experiment &operator=(const Experiment &);
    ~Experiment();

    int number() const;
    QList<QPair<double,QString> > gasSetpoints() const;
    QList<QPair<double,QString> > pressureSetpoints() const;
    QDateTime startTime() const;
    bool isInitialized() const;
    bool isAborted() const;
    bool isDummy() const;

    void setGasSetpoints(QList<QPair<double,QString> > list);
    void addGasSetpoint(double setPoint, QString name);
    void setPressureSetpoints(QList<QPair<double,QString> > list);
    void addPressureSetpoint(double setPoint, QString name);
    void setInitialized();
    void setAborted();
    void setDummy();

private:
    QSharedDataPointer<ExperimentData> data;
};

#endif // EXPERIMENT_H
