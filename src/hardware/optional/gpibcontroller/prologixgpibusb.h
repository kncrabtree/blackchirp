#ifndef PROLOGIXGPIBUSB_H
#define PROLOGIXGPIBUSB_H

#include "gpibcontroller.h"

namespace BC::Key {
static const QString prologixUsb{"prologixGpibUsb"};
static const QString prologixUsbName("Prologix GPIB-USB Controller");
}

class PrologixGpibUsb : public GpibController
{
    Q_OBJECT
public:
    explicit PrologixGpibUsb(QObject *parent = nullptr);

    // HardwareObject interface
protected:
    void initialize() override;
    bool testConnection() override;

    // GpibController interface
public:
    QString queryTerminator() const override;

protected:
    bool readAddress() override;
    bool setAddress(int a) override;
};

#endif // PROLOGIXGPIBUSB_H
