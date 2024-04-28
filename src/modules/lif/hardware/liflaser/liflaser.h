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

    bool d_autoDisable{false};

    // HardwareObject interface
public slots:
    bool hwPrepareForExperiment(Experiment &exp) override final;
    void beginAcquisition() override final;
    void endAcquisition() override final;
};


#ifdef BC_LIFLASER
#include BC_STR(BC_LIFLASER_H)
#endif

#endif // LIFLASER_H
