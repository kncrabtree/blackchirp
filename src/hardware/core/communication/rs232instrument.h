#ifndef RS232INSTRUMENT_H
#define RS232INSTRUMENT_H

#include <hardware/core/communication/communicationprotocol.h>

#include <QtSerialPort/qserialport.h>
#include <QtSerialPort/qserialportinfo.h>

namespace BC::Key::RS232 {
static const QString baud("baudrate");
static const QString id("id");
static const QString dataBits("databits");
static const QString parity("parity");
static const QString stopBits("stopbits");
static const QString flowControl("flowControl");
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
