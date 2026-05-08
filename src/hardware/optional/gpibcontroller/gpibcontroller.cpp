#include <hardware/optional/gpibcontroller/gpibcontroller.h>

GpibController::GpibController(const QString& impl, const QString& label, QObject *parent) :
    HardwareObject(QString(GpibController::staticMetaObject.className()), impl, label, parent)
{
    d_threaded = true;
}

void GpibController::hwReadSettings()
{
    gpibReadSettings();
}

GpibController::~GpibController()
{
    // Release all addresses owned by any hardware objects when controller is destroyed
    QMutexLocker locker(&d_commMutex);
    d_addressOwners.clear();
}

bool GpibController::writeCmd(int address, const QString &cmd)
{
    QMutexLocker locker(&d_commMutex);
    
	if(address != d_currentAddress)
    {
        if(!setAddress(address))
            return false;
    }

    return p_comm->writeCmd(cmd);
}

bool GpibController::writeBinary(int address, const QByteArray &dat)
{
    QMutexLocker locker(&d_commMutex);
    
    if(address != d_currentAddress)
    {
        if(!setAddress(address))
            return false;
    }

    return p_comm->writeBinary(dat);
}

QByteArray GpibController::queryCmd(int address, const QString &cmd, bool suppressError)
{
    QMutexLocker locker(&d_commMutex);

	if(address != d_currentAddress)
    {
        if(!setAddress(address))
            return QByteArray();
    }

    return p_comm->queryCmd(cmd + queryTerminator(), suppressError);
}

bool GpibController::reserveAddress(int address, const QString& ownerKey, const QString& ownerName)
{
    QMutexLocker locker(&d_commMutex);
    
    if (ownerKey.isEmpty()) {
        return false;  // Invalid owner
    }
    
    // Check if address is already in use by someone else
    if (d_addressOwners.contains(address)) {
        const AddressOwner& currentOwner = d_addressOwners.value(address);
        if (currentOwner.key != ownerKey) {
            // Address conflict - someone else owns this address
            return false;
        } else {
            // Same owner is re-reserving the same address - this is OK
            return true;
        }
    }
    
    // Address is available - reserve it
    d_addressOwners.insert(address, {ownerKey, ownerName});
    return true;
}

void GpibController::releaseAddress(int address, const QString& ownerKey)
{
    QMutexLocker locker(&d_commMutex);
    
    if (ownerKey.isEmpty()) {
        return;  // Invalid owner
    }
    
    // Only allow the owner to release their own address
    if (d_addressOwners.contains(address)) {
        const AddressOwner& currentOwner = d_addressOwners.value(address);
        if (currentOwner.key == ownerKey) {
            d_addressOwners.remove(address);
        }
        // If currentOwner.key != ownerKey, silently ignore (not their address to release)
    }
}

bool GpibController::isAddressAvailable(int address) const
{
    QMutexLocker locker(&d_commMutex);
    return !d_addressOwners.contains(address);
}

QList<int> GpibController::getUsedAddresses() const
{
    QMutexLocker locker(&d_commMutex);
    return d_addressOwners.keys();
}

QString GpibController::getAddressOwnerKey(int address) const
{
    QMutexLocker locker(&d_commMutex);
    if (d_addressOwners.contains(address)) {
        return d_addressOwners.value(address).key;
    }
    return QString();
}

QString GpibController::getAddressOwnerName(int address) const
{
    QMutexLocker locker(&d_commMutex);
    if (d_addressOwners.contains(address)) {
        return d_addressOwners.value(address).name;
    }
    return QString();
}

