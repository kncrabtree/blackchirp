#include "virtualgpibcontroller.h"
#include <hardware/core/hardwareregistration.h>

// Register hardware implementation
REGISTER_HARDWARE_META(VirtualGpibController, "Virtual GPIB Controller for Testing")
REGISTER_HARDWARE_SETTINGS(VirtualGpibController)

VirtualGpibController::VirtualGpibController(const QString& label, QObject *parent) :
    GpibController(QString(VirtualGpibController::staticMetaObject.className()), label, parent)
{
    // Initialize communication protocol for testing
    buildCommunication();
    
    // Call bcInitInstrument to complete hardware object initialization
    bcInitInstrument();
}

VirtualGpibController::~VirtualGpibController()
{

}

QString VirtualGpibController::getThreadInfo() const
{
    QThread* currentThread = QThread::currentThread();
    QString threadName = currentThread->objectName();
    if (threadName.isEmpty()) {
        threadName = QString("Thread-0x%1").arg(reinterpret_cast<quintptr>(currentThread), 0, 16);
    }
    return threadName;
}

bool VirtualGpibController::writeCmd(int address, QString cmd)
{
    QMutexLocker locker(&d_commMutex);
    
    if(address != d_currentAddress)
    {
        if(!setAddress(address))
            return false;
    }

    // Simulate successful write
    return true;
}

bool VirtualGpibController::writeBinary(int address, QByteArray dat)
{
    QMutexLocker locker(&d_commMutex);
    
    if(address != d_currentAddress)
    {
        if(!setAddress(address))
            return false;
    }

    // Simulate successful write
    return true;
}

QByteArray VirtualGpibController::queryCmd(int address, QString cmd, bool suppressError)
{
    QMutexLocker locker(&d_commMutex);
    
    if(address != d_currentAddress)
    {
        if(!setAddress(address))
            return QByteArray();
    }

    // Echo back the query with thread info for verification
    QString threadInfo = getThreadInfo();
    QString response = QString("ECHO[%1]:%2").arg(threadInfo).arg(cmd);
    QByteArray result = response.toUtf8();
    
    Q_UNUSED(suppressError)
    return result;
}

bool VirtualGpibController::testConnection()
{
	return true;
}

void VirtualGpibController::initialize()
{
}

bool VirtualGpibController::readAddress()
{
    return true;
}

bool VirtualGpibController::setAddress(int a)
{
	d_currentAddress = a;
    return true;
}

