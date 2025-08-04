#include "hp83712b.h"
#include <hardware/core/hardwareregistration.h>

// Register hardware implementation
REGISTER_HARDWARE_META(HP83712B, "Hewlett-Packard 83712B Synthesized Signal Generator (1-20000 MHz)")

HP83712B::HP83712B(const QString& label, QObject *parent)
    : Clock{1, true, QString(HP83712B::staticMetaObject.className()), label, parent}
{
    setDefault(BC::Key::Clock::minFreq,1.0);
    setDefault(BC::Key::Clock::maxFreq,20000.0);
    setDefault(BC::Key::Clock::lock,false);

    // Communication defaults
    setDefault(BC::Key::Comm::timeout, 500);
    setDefault(BC::Key::Comm::termChar, QString("\n"));

    save();
}


void HP83712B::initializeClock()
{
}

bool HP83712B::testClockConnection()
{
    QByteArray resp = p_comm->queryCmd(QString("*IDN?\n"));
    if(resp.isEmpty())
    {
       d_errorString = QString("Null response to ID query");
        return false;
    }
    if(!resp.contains("83712"))
    {
        d_errorString = QString("ID response invalid. Received: %1").arg(QString(resp));
        return false;
    }

    emit logMessage(QString("ID response: %1").arg(QString(resp)));

    return true;
}

bool HP83712B::setHwFrequency(double freqMHz, int outputIndex)
{
    Q_UNUSED(outputIndex)
    return p_comm->writeCmd(QString(":FREQ %1MHZ\n").arg(freqMHz,0,'f',3));
}

double HP83712B::readHwFrequency(int outputIndex)
{
    Q_UNUSED(outputIndex)
    QByteArray resp = p_comm->queryCmd(QString(":FREQ?\n"));
    return resp.trimmed().toDouble()/1.0e6;
}
