#ifndef RS232INSTRUMENT_H
#define RS232INSTRUMENT_H

#include <hardware/core/communication/communicationprotocol.h>

#include <QtSerialPort/qserialport.h>
#include <QtSerialPort/qserialportinfo.h>

namespace BC::Key::RS232 {
static const QString baud{"baudrate"};
static const QString id{"id"};
static const QString dataBits{"databits"};
static const QString parity{"parity"};
static const QString stopBits{"stopbits"};
static const QString flowControl{"flowControl"};
}

class Rs232Instrument : public CommunicationProtocol
{
	Q_OBJECT
public:
    enum DataBits {
        Data5 = QSerialPort::Data5,
        Data6 = QSerialPort::Data6,
        Data7 = QSerialPort::Data7,
        Data8 = QSerialPort::Data8
    };
    Q_ENUM(DataBits)

    enum Parity {
        NoParity = QSerialPort::NoParity,
        EvenParity = QSerialPort::EvenParity,
        OddParity = QSerialPort::OddParity,
        SpaceParity = QSerialPort::SpaceParity,
        MarkParity = QSerialPort::MarkParity
    };
    Q_ENUM(Parity)

    enum StopBits {
        OneStop = QSerialPort::OneStop,
        OneAndhalfStop = QSerialPort::OneAndHalfStop,
        TwoStop = QSerialPort::TwoStop
    };
    Q_ENUM(StopBits)

    enum FlowControl {
        NoFlowControl = QSerialPort::NoFlowControl,
        HardwareControl = QSerialPort::HardwareControl,
        SoftwareControl = QSerialPort::SoftwareControl
    };
    Q_ENUM(FlowControl)

    explicit Rs232Instrument(QString key, QObject *parent = 0);
    ~Rs232Instrument();


public slots:
    void initialize() override;
    bool testConnection() override;
    bool testManual(QString name, qint32 br);
    
private:
    QSerialPort *p_device;

    
    // CommunicationProtocol interface
public:
    QIODevice *_device() override;
};

#endif // RS232INSTRUMENT_H
