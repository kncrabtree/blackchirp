#ifndef RS232INSTRUMENT_H
#define RS232INSTRUMENT_H

#include <src/hardware/core/communication/communicationprotocol.h>

#include <QtSerialPort/qserialport.h>
#include <QtSerialPort/qserialportinfo.h>

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
