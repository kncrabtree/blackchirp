#include <modules/lif/hardware/liflaser/liflaser.h>

LifLaser::LifLaser(const QString subKey, const QString name, CommunicationProtocol::CommType commType, QObject *parent, bool threaded, bool critical) :
    HardwareObject (BC::Key::LifLaser::key,subKey,name,commType,parent,threaded,critical)
{

}

LifLaser::~LifLaser()
{

}

double LifLaser::readPosition()
{
    double out = readPos();
    if(out < 0.0)
    {
        emit logMessage(QString("Could not read position.").arg(d_name),LogHandler::Error);
        emit hardwareFailure();
    }
    else {
        emit laserPosUpdate(out);
    }

    return out;
}

double LifLaser::setPosition(const double pos)
{
    using namespace BC::Key::LifLaser;
    auto minp = get(minPos,200.0);
    auto maxp = get(maxPos,2000.0);
    if(pos < minp || pos > maxp)
    {
        auto d = get(decimals,2);
        emit logMessage(QString("Requested position (%1 %2) is outside the allowed range of %3 %2 - %4 %2.").arg(pos,0,'f',d).arg(get(units, "nm").toString()).arg(minp,0,'f',d).arg(maxp,0,'f',d),LogHandler::Error);
        emit hardwareFailure();
        return -1.0;
    }

    setPos(pos);

    return readPosition();
}

bool LifLaser::readFlashLamp()
{
    auto out = readFl();
    emit laserFlashlampUpdate(out);
    return out;
}

bool LifLaser::setFlashLamp(bool en)
{
    if(setFl(en))
    {
        readFlashLamp();
        return true;
    }

    return false;
}
