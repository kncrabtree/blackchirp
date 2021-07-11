#ifndef EXPERIMENTOBJECTIVE_H
#define EXPERIMENTOBJECTIVE_H

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

    int d_number{-1};
};

#endif // EXPERIMENTOBJECTIVE_H
