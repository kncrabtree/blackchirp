#include <src/hardware/core/ioboard/ioboard.h>

IOBoard::IOBoard(const QString subKey, const QString name, CommunicationProtocol::CommType commType, QObject *parent, bool threaded, bool critical)  :
    HardwareObject(BC::Key::ioboard, subKey, name, commType, parent, threaded, critical),
    d_numAnalog(0), d_numDigital(0), d_reservedAnalog(0), d_reservedDigital(0)
{

}

IOBoard::~IOBoard()
{

}

void IOBoard::readSettings()
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(d_key);
    s.beginGroup(d_subKey);

    d_numAnalog = qBound(0,s.value(QString("numAnalog"),4).toInt(),16);
    d_numDigital = qBound(0,s.value(QString("numDigital"),16-d_numAnalog).toInt(),16);
    d_reservedAnalog = qMin(d_numAnalog,s.value(QString("reservedAnalog"),0).toInt());
    d_reservedDigital = qMin(d_numDigital,s.value(QString("reservedDigital"),0).toInt());

    s.setValue(QString("numAnalog"),d_numAnalog);
    s.setValue(QString("numDigital"),d_numDigital);
    s.setValue(QString("reservedAnalog"),d_reservedAnalog);
    s.setValue(QString("reservedDigital"),d_reservedDigital);

    s.endGroup();
    s.endGroup();

    s.sync();

    readIOBSettings();
}
