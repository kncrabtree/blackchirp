#include "prologixgpibcontroller.h"
#include <data/settings/hardwarekeys.h>

PrologixGpibController::PrologixGpibController(const QString& impl, const QString& label, 
                                               CommunicationProtocol::CommType commType, 
                                               QObject *parent) :
    GpibController(impl, label, parent)
{
    // Set communication type preference
    setDefault(BC::Key::HW::commType, static_cast<int>(commType));
    
    // Common communication defaults
    setDefault(BC::Key::Comm::timeout, 1000);
    setDefault(BC::Key::Comm::termChar, QString("\n"));

    save();
}

bool PrologixGpibController::testConnection()
{
    QByteArray resp = p_comm->queryCmd(QString("++ver\n"));
    if(resp.isEmpty())
    {
        d_errorString = QString("%1 gave a null response to ID query").arg(d_name);
        return false;
    }
    
    if(!resp.startsWith(expectedIdResponse()))
    {
        d_errorString = QString("%1 response invalid. Received: %2").arg(d_name).arg(QString(resp));
        return false;
    }

    emit logMessage(QString("%1 ID response: %2").arg(d_name).arg(QString(resp)));

    // Send common configuration commands
    p_comm->writeCmd(QString("++auto 0\n"));
    if(shouldSendSaveCfg()) {
        p_comm->writeCmd(QString("++savecfg 0\n"));
    }
    p_comm->writeCmd(QString("++read_tmo_ms 50\n"));

    readAddress();

    return true;
}

void PrologixGpibController::initialize()
{
}

bool PrologixGpibController::readAddress()
{
    bool success = false;
    QByteArray resp = p_comm->queryCmd(QString("++addr\n"));
    d_currentAddress = resp.trimmed().toInt(&success);
    if(!success)
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not read address. Response: %1 (Hex: %2)").arg(QString(resp)).arg(QString(resp.toHex())));
    }
    return success;
}

bool PrologixGpibController::setAddress(int a)
{
    if(!p_comm->writeCmd(QString("++addr %1\n").arg(a)))
        return false;

    if(!readAddress())
        return false;

    if(d_currentAddress != a)
    {
        emit hardwareFailure();
        emit logMessage(QString("Address was not set to %1. Current address: %2").arg(a).arg(d_currentAddress),LogHandler::Error);
        return false;
    }

    return true;
}

QString PrologixGpibController::queryTerminator() const
{
    return QString("++read eoi\n");
}