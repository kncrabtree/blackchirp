#ifndef VIRTUALAWG_H
#define VIRTUALAWG_H

#include "awg.h"

class VirtualAwg : public AWG
{
    Q_OBJECT
public:
    explicit VirtualAwg(QObject *parent = nullptr);
    ~VirtualAwg();

    // HardwareObject interface
public slots:
    void readSettings();
    Experiment prepareForExperiment(Experiment exp);
    void beginAcquisition();
    void endAcquisition();

protected:
    bool testConnection();
    void initialize();

};

#endif // VIRTUALAWG_H
