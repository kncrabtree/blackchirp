#ifndef PROLOGIXGPIBLAN_H
#define PROLOGIXGPIBLAN_H

#include <hardware/optional/gpibcontroller/prologixgpibcontroller.h>

class PrologixGpibLan : public PrologixGpibController
{
    Q_OBJECT
public:
    explicit PrologixGpibLan(const QString& label, QObject *parent = nullptr);

protected:
    // PrologixGpibController interface
    QString expectedIdResponse() const override;
    bool shouldSendSaveCfg() const override;
};

#endif // PROLOGIXGPIBLAN_H
