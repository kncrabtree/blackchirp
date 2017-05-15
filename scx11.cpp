#include "scx11.h"

Scx11::Scx11(QObject *parent) : MotorController(parent)
{
    d_subKey = QString("scx11");
    d_prettyName = QString("Motor controller SCX11");
    d_commType = CommunicationProtocol::Rs232;

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(d_key);
    s.beginGroup(d_subKey);
    int d_xId = s.value(QString("xId"),QString("0")).toInt();
    int d_yId = s.value(QString("yId"),QString("1")).toInt();
    int d_zId = s.value(QString("zId"),QString("2")).toInt();
    s.setValue(QString("xId"),d_xId);
    s.setValue(QString("yId"),d_yId);
    s.setValue(QString("zId"),d_zId);

    double xHomeOffset = s.value(QString("xHomeOffset"),0).toDouble(); //total: about 77
    double yHomeOffset = s.value(QString("yHomeOffset"),80).toDouble(); //total: 162.516
    double zHomeOffset = s.value(QString("zHomeOffset"),200).toDouble(); //total: about 397
    s.setValue(QString("xHomeOffset"),xHomeOffset);
    s.setValue(QString("yHomeOffset"),yHomeOffset);
    s.setValue(QString("zHomeOffset"),zHomeOffset);

    double xMin = s.value(QString("xMin"),-100.0).toDouble();
    double xMax = s.value(QString("xMax"),100.0).toDouble();
    double yMin = s.value(QString("yMin"),-100.0).toDouble();
    double yMax = s.value(QString("yMax"),100.0).toDouble();
    double zMin = s.value(QString("zMin"),-100.0).toDouble();
    double zMax = s.value(QString("zMax"),100.0).toDouble();

    d_xRestingPos = s.value(QString("xRest"),xMin).toDouble();
    d_yRestingPos = s.value(QString("yRest"),yMin).toDouble();
    d_zRestingPos = s.value(QString("zRest"),zMin).toDouble();

    s.setValue(QString("xMin"),xMin);
    s.setValue(QString("xMax"),xMax);
    s.setValue(QString("yMin"),yMin);
    s.setValue(QString("yMax"),yMax);
    s.setValue(QString("zMin"),zMin);
    s.setValue(QString("zMax"),zMax);
    s.setValue(QString("xRest"),d_xRestingPos);
    s.setValue(QString("yRest"),d_yRestingPos);
    s.setValue(QString("zRest"),d_zRestingPos);

    d_xRange = qMakePair(xMin,xMax);
    d_yRange = qMakePair(yMin,yMax);
    d_zRange = qMakePair(zMin,zMax);
    s.endGroup();
    s.endGroup();

}



bool Scx11::testConnection()
{
    if(!p_comm->testConnection())
    {
        emit connected(false,QString("RS232 error"));
        return false;
    }

    QByteArray resp = p_comm->queryCmd(QString("VER\n"));

    if(resp.isEmpty())
    {
        emit connected(false,QString("Could not receive responce to version query."));
        return false;
    }

    if(resp.startsWith("VER"))
    {
        p_comm->writeCmd(QString("ECHO=0\n"));
        QByteArray t;
        while(p_comm->device()->waitForReadyRead(1000))
        {
            t.append(p_comm->device()->readAll());
            if(t.endsWith(">"))
                break;
        }
        resp = p_comm->queryCmd(QString("VER\n"));
    }

    if(!resp.startsWith("SCX11"))
    {
        emit connected(false, QString("Could not connect to SCX11. ID response: %1").arg(QString(resp.trimmed())));
        return false;
    }

    emit logMessage(QString("ID response: %1").arg(QString(resp.trimmed())));
    emit connected();
    p_limitTimer->start();
    d_nextRead = 0;
    return true;
}

void Scx11::initialize()
{
    p_comm->initialize();
    p_comm->setReadOptions(1000,true,QByteArray(">"));
    p_limitTimer->stop();
    testConnection();


    // need to set default velocity and acceleration time here.

}

void Scx11::beginAcquisition()
{
}

void Scx11::endAcquisition()
{
}

void Scx11::readTimeData()
{
}

void Scx11::moveToPosition(double x, double y, double z)
{
    BlackChirp::MotorAxis axis;
    bool done;

    if (abs(d_xPos-x) >= 0.001)
    {
        axis = BlackChirp::MotorAxis::MotorX;
        done = Scx11::moveAxis(axis, x);
        if (!done)
        {
            emit hardwareFailure();
            return;
        }

    }
    if (abs(d_yPos-y) >= 0.001)
    {
        axis = BlackChirp::MotorAxis::MotorY;
        done = Scx11::moveAxis(axis, y);
        if (!done)
        {
            emit hardwareFailure();
            return;
        }
    }
    if (abs(d_zPos-z) >= 0.001)
    {
        axis = BlackChirp::MotorAxis::MotorZ;
        done = Scx11::moveAxis(axis, z);
        if (!done)
        {
            emit hardwareFailure();
            return;
        }
    }
    return;
}

bool Scx11::prepareForMotorScan(const MotorScan ms)
{
    Q_UNUSED(ms)
    return true;
}

void Scx11::moveToRestingPos()
{
    Scx11::moveToPosition(d_xRestingPos, d_yRestingPos, d_zRestingPos);
}

void Scx11::checkLimit()
{
    BlackChirp::MotorAxis axis;

    if(d_nextRead == 0)
    {
        axis = BlackChirp::MotorAxis::MotorX;
    }
    else if(d_nextRead == 1)
    {
        axis = BlackChirp::MotorAxis::MotorY;
    }
    else if(d_nextRead == 2)
    {
        axis = BlackChirp::MotorAxis::MotorY;
    }

    Scx11::checkLimitOneAxis(axis);

    d_nextRead += 1;
    if(d_nextRead == 3)
    {
        d_nextRead = 0;
    }
}

bool Scx11::moveAxis(BlackChirp::MotorAxis axis, double pos)
{
    int id;
    QString axisName;

    switch(axis)
    {
    case BlackChirp::MotorX:
        id = d_xId;
        axisName = QString("X");
        break;
    case BlackChirp::MotorY:
        id = d_yId;
        axisName = QString("Y");
        break;
    case BlackChirp::MotorZ:
        id = d_zId;
        axisName = QString("Z");
        break;
    }

    p_comm->writeCmd(QString("@%1\n").arg(id));
    p_comm->writeCmd(QString("MA %1\n").arg(pos,0,'f',3));

    QByteArray resp;
    bool done = false;
    while(!done)
    {
        p_comm->device()->waitForReadyRead(50);
        resp = p_comm->queryCmd(QString("SIGMOVE\n"));
        if(resp.contains("0"))
        {
            done = true;
        }

        if(resp.isEmpty())
        {
            break;
        }
    }
    if(!done)
    {
        emit logMessage(QString("Error occured during motion of axis %1. Sequence aborted.").arg(axisName));
        return false;
    }

    emit posUpdate(axis, pos);
    return true;
}

void Scx11::checkLimitOneAxis(BlackChirp::MotorAxis axis)
{
    int id;

    switch(axis)
    {
    case BlackChirp::MotorX:
        id = d_xId;
        break;
    case BlackChirp::MotorY:
        id = d_yId;
        break;
    case BlackChirp::MotorZ:
        id = d_zId;
        break;
    }

    bool sigPositive, sigNegative;

    p_comm->writeCmd(QString("@%1\n").arg(id));
    QByteArray resp = p_comm->queryCmd(QString("SIGLSP\n"));

    if(resp.contains("0"))
    {
        sigPositive = false;
    }
    else if(resp.contains("1"))
    {
        sigPositive = true;
    }
    else
    {
        emit hardwareFailure();
        return;
    }

    resp = p_comm->queryCmd(QString("SIGLSN\n"));

    if(resp.contains("0"))
    {
        sigNegative = false;
    }
    else if(resp.contains("1"))
    {
        sigNegative = true;
    }
    else
    {
        emit hardwareFailure();
        return;
    }

    emit limitStatus(axis, sigNegative, sigPositive);
    return;
}
