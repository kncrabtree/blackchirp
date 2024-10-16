#pragma once //This is a temporary workaround to avoid a bug with syntax highlighting in Qt Creator 4.8. It is fixed in 4.10
#ifndef VIRTUALLIFLASER_H
#define VIRTUALLIFLASER_H

#include <modules/lif/hardware/liflaser/liflaser.h>

namespace BC::Key {
static const QString vLifLaser("Virtual LIF Laser");
}

class VirtualLifLaser : public LifLaser
{
    Q_OBJECT
public:
    VirtualLifLaser(QObject *parent = nullptr);

protected:
    void initialize() override;
    bool testConnection() override;

    // LifLaser interface
private:
    double readPos() override;
    void setPos(double pos) override;
    bool readFl() override;
    bool setFl(bool en) override;

    double d_pos;
    bool d_fl;
};

#endif // VIRTUALLIFLASER_H
