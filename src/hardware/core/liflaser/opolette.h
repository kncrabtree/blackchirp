#ifndef OPOLETTE_H
#define OPOLETTE_H

#include "liflaser.h"

class Opolette : public LifLaser
{
    Q_OBJECT
public:
    explicit Opolette(const QString& label, QObject *parent = nullptr);

    // HardwareObject interface
protected:
    void initialize();
    bool testConnection();

    // LifLaser interface
private:
    double readPos();
    void setPos(double pos);
    bool readFl();
    bool setFl(bool en);
};

using LifLaserHardware = Opolette;

#endif // OPOLETTE_H
