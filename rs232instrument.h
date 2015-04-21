#ifndef RS232INSTRUMENT_H
#define RS232INSTRUMENT_H

#include "communicationprotocol.h"

#include <QtSerialPort/qserialport.h>
#include <QtSerialPort/qserialportinfo.h>

//QT_USE_NAMESPACE_SERIALPORT

//NOTE: Finding device information on Linux.
//Device mapping is not static on Linux. Serial devices can be mapped to different places dependong on the order in which they're plugged in or activated
//If the system is rebooted, it's possible that the mappings will change, and communication might fail
//To see the devices and their mappings, run ls -l /dev/serial/by-id or /dev/serial/by-path and look at the output.
//If the serial deivces are from different manufacturers, you can tell what's what.
//If not (for example, you have multiple USB-RS232 converters from the same manufactuer), you can disconnect them one at a time
//run those commands, then plug it back in, rerunning the command to figure out which it is.
//Existing names aren't reassigned when a new device is connected, so you should see one entry disappear and then reappear.


class Rs232Instrument : public CommunicationProtocol
{
	Q_OBJECT
public:
    explicit Rs232Instrument(QString key, QString subKey, QObject *parent = 0);
    ~Rs232Instrument();
    bool writeCmd(QString cmd);
    QByteArray queryCmd(QString cmd);
    QSerialPort *d_sp;


public slots:
	void initialize();
	bool testConnection();

};

#endif // RS232INSTRUMENT_H
