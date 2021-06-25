#ifndef RS232INSTRUMENT_H
#define RS232INSTRUMENT_H

#include <hardware/core/communication/communicationprotocol.h>

#include <QtSerialPort/qserialport.h>
#include <QtSerialPort/qserialportinfo.h>

namespace BC::Key {
static const QString rs232baud("baudrate");
static const QString rs232id("id");
static const QString rs232dataBits("databits");
static const QString rs232parity("parity");
static const QString rs232stopBits("stopbits");
static const QString rs232flowControl("flowControl");
}

class Rs232Instrument : public CommunicationProtocol
{
	Q_OBJECT
public:
    explicit Rs232Instrument(QString key, QObject *parent = 0);
    ~Rs232Instrument();


public slots:
    void initialize() override;
    bool testConnection() override;

};

#endif // RS232INSTRUMENT_H
