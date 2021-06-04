#pragma once //This is a temporary workaround to avoid a bug with syntax highlighting in Qt Creator 4.8. It is fixed in 4.10
#ifndef VIRTUALLIFLASER_H
#define VIRTUALLIFLASER_H

#include <src/modules/lif/hardware/liflaser/liflaser.h>

class VirtualLifLaser : public LifLaser
{
    Q_OBJECT
public:
    VirtualLifLaser(QObject *parent = nullptr);

    // HardwareObject interface
public slots:
    void readSettings() override;
    void sleep(bool b) override;

protected:
    void initialize() override;
    bool testConnection() override;

    // LifLaser interface
private:
    double readPos() override;
    void setPos(double pos) override;

    double d_pos;
};

#endif // VIRTUALLIFLASER_H
