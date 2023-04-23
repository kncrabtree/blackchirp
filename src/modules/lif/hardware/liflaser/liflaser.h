#ifndef LIFLASER_H
#define LIFLASER_H

#include <hardware/core/hardwareobject.h>

namespace BC::Key::LifLaser {
static const QString key{"LifLaser"};
static const QString units{"units"};
static const QString decimals{"decimals"};
static const QString minPos{"minPos"};
static const QString maxPos{"maxPos"};
}

class LifLaser : public HardwareObject
{
    Q_OBJECT
public:
    LifLaser(const QString subKey, const QString name, CommunicationProtocol::CommType commType, QObject *parent = nullptr, bool threaded=false,bool critical=true);
    ~LifLaser() override;

signals:
    void laserPosUpdate(double);
    void laserFlashlampUpdate(bool);

public slots:
    double readPosition();
    double setPosition(double pos);
    bool readFlashLamp();
    bool setFlashLamp(bool en);

private:
    virtual double readPos() =0;
    virtual void setPos(double pos) =0;
    virtual bool readFl() =0;

    //This function should return whether setting was successful, not whether it's enabled
    virtual bool setFl(bool en) =0;
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
