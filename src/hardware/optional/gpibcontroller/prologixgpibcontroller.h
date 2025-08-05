#ifndef PROLOGIXGPIBCONTROLLER_H
#define PROLOGIXGPIBCONTROLLER_H

#include <hardware/optional/gpibcontroller/gpibcontroller.h>

class PrologixGpibController : public GpibController
{
    Q_OBJECT
public:
    explicit PrologixGpibController(const QString& impl, const QString& label, 
                                   CommunicationProtocol::CommType commType, 
                                   QObject *parent = nullptr);

    // GpibController interface
    QString queryTerminator() const override;

protected:
    // HardwareObject interface
    bool testConnection() override;
    void initialize() override;

    // GpibController interface
    bool readAddress() override;
    bool setAddress(int a) override;

    // Virtual methods for derived classes to customize behavior
    virtual QString expectedIdResponse() const = 0;
    virtual bool shouldSendSaveCfg() const = 0;
};

#endif // PROLOGIXGPIBCONTROLLER_H