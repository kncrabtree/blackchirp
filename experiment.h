#ifndef EXPERIMENT_H
#define EXPERIMENT_H

#include <QSharedDataPointer>

class ExperimentData;

class Experiment
{
public:
    Experiment();
    Experiment(const Experiment &);
    Experiment &operator=(const Experiment &);
    ~Experiment();

    int number() const;

private:
    QSharedDataPointer<ExperimentData> data;
};

#endif // EXPERIMENT_H
