#ifndef VIRTUALLIFSCOPE_H
#define VIRTUALLIFSCOPE_H

#include "lifscope.h"

class VirtualLifScope : public LifScope
{
    Q_OBJECT
public:
    VirtualLifScope(QObject *parent = nullptr);
    ~VirtualLifScope();


public slots:
    // HardwareObject interface
    void readSettings();
    Experiment prepareForExperiment(Experiment exp);
    void beginAcquisition();
    void endAcquisition();

    // LifScope interface
    void setLifVScale(double scale);
    void setRefVScale(double scale);
    void setHorizontalConfig(double sampleRate, int recLen);
    void queryScope();
    void setRefEnabled(bool en);

protected:
    bool testConnection();
    void initialize();

};

#endif // VIRTUALLIFSCOPE_H
