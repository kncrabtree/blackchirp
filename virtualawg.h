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
    void readSettings() override;

protected:
    bool testConnection() override;
    void initialize() override;

};

#endif // VIRTUALAWG_H
