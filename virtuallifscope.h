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
    void readSettings() override;

    // LifScope interface
    void setLifVScale(double scale) override;
    void setRefVScale(double scale) override;
    void setHorizontalConfig(double sampleRate, int recLen) override;
    void queryScope() override;
    void setRefEnabled(bool en) override;

protected:
    bool testConnection() override;
    void initialize() override;

};

#endif // VIRTUALLIFSCOPE_H
