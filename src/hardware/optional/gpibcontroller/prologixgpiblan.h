#ifndef PROLOGIXGPIBLAN_H
#define PROLOGIXGPIBLAN_H

#include <src/hardware/optional/gpibcontroller/gpibcontroller.h>

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
