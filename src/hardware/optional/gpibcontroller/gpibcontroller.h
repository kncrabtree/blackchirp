#ifndef GPIBCONTROLLER_H
#define GPIBCONTROLLER_H

#include <hardware/core/hardwareobject.h>
#include <QMutex>
#include <QHash>

class GpibController : public HardwareObject
{
	Q_OBJECT
public:
    GpibController(const QString& impl, const QString& label, QObject *parent = nullptr);
    virtual ~GpibController();

	bool writeCmd(int address, const QString &cmd);
    bool writeBinary(int address, const QByteArray &dat);
    QByteArray queryCmd(int address, const QString &cmd, bool suppressError=false);
    virtual QString queryTerminator() const { return QString(); }
    
    // GPIB Address Management
    bool reserveAddress(int address, const QString& ownerKey, const QString& ownerName);
    void releaseAddress(int address, const QString& ownerKey);
    bool isAddressAvailable(int address) const;
    QList<int> getUsedAddresses() const;
    QString getAddressOwnerKey(int address) const;
    QString getAddressOwnerName(int address) const;

protected:
    virtual bool readAddress() =0;
    virtual bool setAddress(int a) =0;

	int d_currentAddress;
    mutable QMutex d_commMutex;  // mutable so const methods can lock it
    
    struct AddressOwner {
        QString key;
        QString name;
    };
    QHash<int, AddressOwner> d_addressOwners;  // Track which device owns each address
};

#endif // GPIBCONTROLLER_H
