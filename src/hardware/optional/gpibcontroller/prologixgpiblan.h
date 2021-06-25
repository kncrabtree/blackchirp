#ifndef PROLOGIXGPIBLAN_H
#define PROLOGIXGPIBLAN_H

#include <hardware/optional/gpibcontroller/gpibcontroller.h>

namespace BC::Key {
static const QString prologix("prologixGpibLan");
static const QString prologixName("Prologix GPIB-LAN Controller");
}

class PrologixGpibLan : public GpibController
{
    Q_OBJECT
public:
    explicit PrologixGpibLan(QObject *parent = nullptr);

    // GpibController interface
public:
    QString queryTerminator() const override;

protected:
    // HardwareObject interface
    bool testConnection() override;
    void initialize() override;

    // GpibController interface
    bool readAddress() override;
    bool setAddress(int a) override;


};

#endif // PROLOGIXGPIBLAN_H
