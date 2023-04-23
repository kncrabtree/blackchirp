#ifndef OPOLETTE_H
#define OPOLETTE_H

#include "liflaser.h"

namespace BC::Key::LifLaser {
static const QString opo("opolette");
static const QString opoName("OPOTEK Opolette");
}

class Opolette : public LifLaser
{
    Q_OBJECT
public:
    explicit Opolette(QObject *parent = nullptr);

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

    // HardwareObject interface
public slots:
    void beginAcquisition();
    void endAcquisition();
};

using LifLaserHardware = Opolette;

#endif // OPOLETTE_H
