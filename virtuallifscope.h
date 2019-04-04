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
    bool testConnection();
    void initialize();
    Experiment prepareForExperiment(Experiment exp);
    void beginAcquisition();
    void endAcquisition();
    void readTimeData();

    // LifScope interface
    void setLifVScale(double scale);
    void setRefVScale(double scale);
    void setHorizontalConfig(double sampleRate, int recLen);
    void queryScope();
    void setRefEnabled(bool en);
};

#endif // VIRTUALLIFSCOPE_H
