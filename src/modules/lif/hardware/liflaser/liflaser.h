#ifndef LIFLASER_H
#define LIFLASER_H

#include <hardware/core/hardwareobject.h>

namespace BC::Key {
static const QString lifLaser{"LifLaser"};
static const QString lifLaserUnits{"units"};
}

class LifLaser : public HardwareObject
{
    Q_OBJECT
public:
    LifLaser(const QString subKey, const QString name, CommunicationProtocol::CommType commType, QObject *parent = nullptr, bool threaded=false,bool critical=true);
    LifLaser(LifLaser &) =delete;
    LifLaser(LifLaser &&) =delete;
    LifLaser& operator= (const LifLaser &) =delete;
    LifLaser& operator= (const LifLaser &&) =delete;
    ~LifLaser() override;

signals:
    void laserPosUpdate(double);

public slots:
    double readPosition();
    double setPosition(double pos);

private:
    virtual double readPos() =0;
    virtual void setPos(double pos) =0;

protected:
    double d_minPos=0.0, d_maxPos=0.0;
    int d_decimals = 2;
    QString d_units;
};


#if BC_LIFLASER == 1
#include "opolette.h"
class Opolette;
typedef Opolette LifLaserHardware;
#else
#include "virtualliflaser.h"
class VirtualLifLaser;
using LifLaserHardware = VirtualLifLaser;
#endif

#endif // LIFLASER_H
