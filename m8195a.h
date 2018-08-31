#ifndef M8195A_H
#define M8195A_H

#include "awg.h"

class M8195A : public AWG
{
    Q_OBJECT
public:
    explicit M8195A(QObject *parent=nullptr);

    // HardwareObject interface
public slots:
    virtual bool testConnection();
    virtual void initialize();
    virtual Experiment prepareForExperiment(Experiment exp);
    virtual void beginAcquisition();
    virtual void endAcquisition();
    virtual void readTimeData();
};

#endif // M8195A_H
