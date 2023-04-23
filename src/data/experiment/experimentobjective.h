#ifndef EXPERIMENTOBJECTIVE_H
#define EXPERIMENTOBJECTIVE_H

#include <QVariant>

/*!
 * \brief Abstract base class for components of an experiment
 */
class ExperimentObjective
{
public:
    ExperimentObjective();

    virtual bool initialize() =0;
    virtual bool advance() =0;
    virtual void hwReady() =0;
    virtual int perMilComplete() const =0;
    virtual bool indefinite() const =0;
    virtual bool isComplete() const =0;
    virtual bool abort() =0;
    virtual void cleanupAndSave() =0;
    virtual QString objectiveKey() const =0;
    virtual QVariant objectiveData() const { return QString(""); }

    int d_number{-1};
    QString d_path{""};
    QString d_errorString{""};
    bool d_processingPaused{false};
};

#endif // EXPERIMENTOBJECTIVE_H
