#ifndef PROLOGIXGPIBUSB_H
#define PROLOGIXGPIBUSB_H

#include <hardware/optional/gpibcontroller/prologixgpibcontroller.h>

class PrologixGpibUsb : public PrologixGpibController
{
    Q_OBJECT
public:
    explicit PrologixGpibUsb(const QString& label, QObject *parent = nullptr);

protected:
    // PrologixGpibController interface
    QString expectedIdResponse() const override;
    bool shouldSendSaveCfg() const override;
};

#endif // PROLOGIXGPIBUSB_H
