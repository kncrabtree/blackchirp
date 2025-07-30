#include "gpibinstrument.h"

#include <hardware/optional/gpibcontroller/gpibcontroller.h>

GpibInstrument::GpibInstrument(QString key, GpibController *c, QObject *parent) :
    CommunicationProtocol(key,parent), p_controller(c), d_address(-1)
{
    // Store the parent's key for address management (avoids pointer access during destruction)
    if (auto* hwParent = qobject_cast<HardwareObject*>(parent)) {
        d_ownerKey = hwParent->d_key;
    }
    
    // Connect to controller's destroyed signal for lifecycle management
    if (p_controller) {
        connect(p_controller, &QObject::destroyed, this, &GpibInstrument::onControllerDestroyed);
    }
}

GpibInstrument::~GpibInstrument()
{
    // Release GPIB address when the instrument is destroyed
    if (p_controller && d_address >= 0 && !d_ownerKey.isEmpty()) {
        p_controller->releaseAddress(d_address, d_ownerKey);
    }
}

void GpibInstrument::onControllerDestroyed()
{
    // Controller has been destroyed - invalidate pointer and emit failure
    p_controller = nullptr;
    emit hardwareFailure();
}

void GpibInstrument::setAddress(int a)
{
    if (!p_controller) {
        return;
    }
    
    HardwareObject* hwParent = qobject_cast<HardwareObject*>(parent());
    if (!hwParent) {
        return;  // Invalid parent
    }
    
    // Release old address if we had one
    if (d_address >= 0 && !d_ownerKey.isEmpty()) {
        p_controller->releaseAddress(d_address, d_ownerKey);
    }
    
    // Try to reserve the new address
    if (a >= 0 && !d_ownerKey.isEmpty() && p_controller->reserveAddress(a, d_ownerKey, hwParent->d_name)) {
        d_address = a;
    } else {
        d_address = -1;  // Failed to reserve, mark as invalid
    }
}

int GpibInstrument::address() const
{
	return d_address;
}



bool GpibInstrument::writeCmd(QString cmd)
{
    if (!p_controller) {
        return false;
    }
    return p_controller->writeCmd(d_address,cmd);
}

bool GpibInstrument::writeBinary(QByteArray dat)
{
    if (!p_controller) {
        return false;
    }
    return p_controller->writeBinary(d_address,dat);
}

QByteArray GpibInstrument::queryCmd(QString cmd, bool suppressError)
{
    if (!p_controller) {
        return QByteArray();
    }
    return p_controller->queryCmd(d_address,cmd,suppressError);
}

void GpibInstrument::initialize()
{
}

bool GpibInstrument::testConnection()
{
    if (!p_controller) {
        return false;
    }
    
    HardwareObject* hwParent = qobject_cast<HardwareObject*>(parent());
    if (!hwParent) {
        setErrorString("Invalid parent object - not a HardwareObject");
        return false;
    }
    
    SettingsStorage s(d_key,SettingsStorage::Hardware);
    int requestedAddress = s.getGroupValue<int>(BC::Key::Comm::gpib, BC::Key::GPIB::gpibAddress, 1);
    
    // Try to reserve the address
    if (!p_controller->reserveAddress(requestedAddress, d_ownerKey, hwParent->d_name)) {
        // Address conflict - someone else is already using this address
        QString conflictOwnerName = p_controller->getAddressOwnerName(requestedAddress);
        setErrorString(QString("GPIB address %1 is already in use by %2")
                       .arg(requestedAddress)
                       .arg(conflictOwnerName.isEmpty() ? QString("unknown device") : conflictOwnerName));
        return false;
    }
    
    // Address reservation successful - update our address
    // Release old address first if we had one
    if (d_address >= 0 && d_address != requestedAddress && !d_ownerKey.isEmpty()) {
        p_controller->releaseAddress(d_address, d_ownerKey);
    }
    d_address = requestedAddress;
    
    // Here we could add actual hardware communication tests
    // For now, just return true if we successfully reserved the address
    return true;
}


QIODevice *GpibInstrument::_device()
{
    return nullptr;
}
