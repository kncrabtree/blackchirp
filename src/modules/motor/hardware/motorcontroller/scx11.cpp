#include "scx11.h"

Scx11::Scx11(QObject *parent) : MotorController(parent), d_idle(true)
{
    d_subKey = QString("scx11");
    d_prettyName = QString("Motor controller SCX11");
    d_commType = CommunicationProtocol::Rs232;

}

void Scx11::readSettings()
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(d_key);
    s.beginGroup(d_subKey);
    s.beginReadArray(QString("channels"));

    for(int i=0; i<3; i++)
    {
        s.setArrayIndex(i);
        AxisInfo ai;
        ai.id = s.value(QString("id"),i+1).toInt();
        ai.offset = s.value(QString("offset"),100).toDouble();
        ai.min = s.value(QString("min"),-100.0).toDouble();
        ai.max = s.value(QString("max"),100.0).toDouble();
        ai.rest= s.value(QString("rest"),ai.min).toDouble();
        ai.axis = axisIndex(i);
        ai.name = axisName(ai.axis);
        ai.moving = false;
        ai.nextPos = ai.rest;
        d_channels.append(ai);
    }

    s.endArray();

    s.beginWriteArray(QString("channels"));
    for(int i=0; i<d_channels.size();i++)
    {
        s.setArrayIndex(i);
        s.setValue(QString("id"),d_channels.at(i).id);
        s.setValue(QString("offset"),d_channels.at(i).offset);
        s.setValue(QString("min"),d_channels.at(i).min);
        s.setValue(QString("max"),d_channels.at(i).max);
        s.setValue(QString("rest"),d_channels.at(i).rest);
    }

    s.endArray();
    s.endGroup();
    s.endGroup();

    d_xRange = qMakePair(d_channels.at(0).min,d_channels.at(0).max);
    d_yRange = qMakePair(d_channels.at(1).min,d_channels.at(1).max);
    d_zRange = qMakePair(d_channels.at(2).min,d_channels.at(2).max);
}



bool Scx11::testConnection()
{
    QByteArray resp;

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(d_key);
    s.beginGroup(d_subKey);
    s.beginReadArray(QString("channels"));

    for(int i = 0; i < d_channels.size(); i++)
    {
        resp = p_comm->queryCmd(QString("@%1@%1\n").arg(d_channels.at(i).id));
        if(resp.isEmpty())
        {
            d_errorString = QString("Could not communicate with %1 axis").arg(d_channels.at(i).name);
            return false;
        }

        resp = p_comm->queryCmd(QString("VER\n"));
        if(resp.isEmpty())
        {
            d_errorString = QString("Could not get version info from %1 axis").arg(d_channels.at(i).name);
            return false;
        }

        if(resp.startsWith("VER"))
        {
            QByteArray t = p_comm->queryCmd(QString("ECHO=0\n"));
            if(t.isEmpty())
            {
                d_errorString = QString("Could not disable echo on %1 axis").arg(d_channels.at(i).name);
                return false;
            }

            resp = p_comm->queryCmd(QString("VER\n"));
            if(resp.isEmpty())
            {
                d_errorString = QString("Could not get version info from %1 axis").arg(d_channels.at(i).name);
                return false;
            }
        }
        emit logMessage(QString("ID response: %1").arg(QString(resp.trimmed())));

        if(!resp.startsWith("SCX11"))
        {
            d_errorString =  QString("Could not connect to SCX11. ID response: %1").arg(QString(resp.trimmed()));
            return false;
        }

        QByteArray t = p_comm->queryCmd(QString("VERBOSE=1\n"));
        if(t.isEmpty())
        {
            d_errorString = QString("Could not enable verbose mode on %1 axis").arg(d_channels.at(i).name);
            return false;
        }


        //read negative limit
        resp = p_comm->queryCmd(QString("SIGLSN\n"));
        if(!resp.contains("=1"))
        {
            d_errorString = QString("%1 axis is not at its negative limit. Move it there manually and reconnect.").arg(d_channels.at(i).name);
            return false;
        }

        //set offset
        s.setArrayIndex(i);
        double homeOffset = s.value(QString("offset"),0.0).toDouble();

        t = p_comm->queryCmd(QString("PC=%1\n").arg(-homeOffset,0,'f',3));
        if(t.isEmpty())
        {
            d_errorString = QString("Could not set initial home offset on %1 axis").arg(d_channels.at(i).axis);
            return false;
        }

    }

    s.endArray();
    s.endGroup();
    s.endGroup();


    d_idle = true;
    if(!readCurrentPosition())
    {
        d_errorString = QString("Could not read current position.");
        return false;
    }

    //set speed of Z axis
    p_comm->writeCmd(QString("@3@3\n"));
    p_comm->writeCmd(QString("VR 10\n"));


    p_limitTimer->start();
    d_nextRead = 0;
    return true;
}

void Scx11::initialize()
{
    p_comm->setReadOptions(1000,true,QByteArray(">"));

    p_motionTimer = new QTimer(this);
    p_motionTimer->setInterval(50);

    connect(this,&Scx11::hardwareFailure,p_limitTimer,&QTimer::stop);
    connect(this,&Scx11::hardwareFailure,p_motionTimer,&QTimer::stop);
    connect(p_motionTimer,&QTimer::timeout,this,&Scx11::checkMotion);

}

bool Scx11::moveToPosition(double x, double y, double z)
{
    QList<QPair<double,double>> positions{ qMakePair(d_xPos,x), qMakePair(d_yPos,y), qMakePair(d_zPos,z) };

    if(!d_idle)
    {
        if(p_motionTimer->isActive())
            p_motionTimer->stop();

        //abort current motion
        for(int i=0; i<d_channels.size(); i++)
        {
            AxisInfo ai = d_channels.at(i);
            QByteArray resp = p_comm->queryCmd(QString("@%1@%1\n").arg(ai.id));
            resp = p_comm->queryCmd(QString("HSTOP\n"));
            if(resp.isEmpty())
            {
                emit hardwareFailure();
                emit logMessage(QString("Could not abort current motion to move to next position (%1, %2, %3)").arg(x,0,'f',3).arg(y,0,'f',3).arg(z,0,'f',3),BlackChirp::LogError);
                return false;
            }
        }
    }

    for(int i=0; i<d_channels.size() && i<positions.size(); i++)
    {
        double cPos = positions.at(i).first;
        double nPos = positions.at(i).second;

        if(qAbs(cPos - nPos) >= 0.001)
        {
            d_channels[i].moving = true;
            d_channels[i].nextPos = nPos;
            AxisInfo ai = d_channels.at(i);
            QByteArray resp = p_comm->queryCmd(QString("@%1@%1\n").arg(ai.id));
            resp = p_comm->queryCmd(QString("MA %1\n").arg(ai.nextPos,0,'f',3));
            if(!p_motionTimer->isActive())
                p_motionTimer->start();
            d_idle = false;
        }
    }

    return true;
}

void Scx11::moveToRestingPos()
{
    Scx11::moveToPosition(d_channels.at(0).rest, d_channels.at(1).rest, d_channels.at(2).rest);
}

void Scx11::checkLimit()
{
    checkLimitOneAxis(d_channels.at(d_nextRead).axis);

    d_nextRead += 1;
    d_nextRead %= d_channels.size();
}

bool Scx11::readCurrentPosition()
{
    for(int i=0; i<d_channels.size(); i++)
    {
        int id = d_channels.at(i).id;
        QByteArray resp = p_comm->queryCmd(QString("@%1@%1\n").arg(id));
        if(resp.isEmpty())
        {
            emit hardwareFailure();
            emit logMessage(QString("Could not read position for %1 axis").arg(d_channels.at(i).name),BlackChirp::LogError);
            return false;
        }

        resp = p_comm->queryCmd(QString("PC\n"));
        if(resp.isEmpty() || !resp.contains('=') || !resp.contains('m'))
        {
            emit hardwareFailure();
            emit logMessage(QString("Could not read position for %1 axis").arg(d_channels.at(i).name),BlackChirp::LogError);
            return false;
        }
        int f = resp.indexOf('=');
        int l = resp.indexOf('m',f);
        double num = resp.mid(f+1,l-f-1).trimmed().toDouble();
        emit posUpdate(d_channels.at(i).axis, num);
        switch (d_channels.at(i).axis) {
        case BlackChirp::MotorX:
            d_xPos = num;
            break;
        case BlackChirp::MotorY:
            d_yPos = num;
            break;
        case BlackChirp::MotorZ:
            d_zPos = num;
            break;
        default:
            break;
        }
    }

    return true;
}

void Scx11::checkMotion()
{
    if(d_idle)
        return;

    for(int i=0; i<d_channels.size(); i++)
    {
        AxisInfo ai = d_channels.at(i);
        if(!ai.moving)
            continue;

        QByteArray resp = p_comm->queryCmd(QString("@%1@%1\n").arg(ai.id));
        resp = p_comm->queryCmd(QString("SIGMOVE\n"));
        if(resp.isEmpty())
        {
            p_motionTimer->stop();
            emit motionComplete(false);
            emit hardwareFailure();
            emit logMessage(QString("Error occured during motion of %1 axis. Sequence aborted.").arg(ai.name),BlackChirp::LogError);
            d_idle = true;
            return;

        }
        if(resp.contains("=0"))
        {
            d_channels[i].moving = false;
            if(!moving())
            {
                d_idle = true;
                emit motionComplete();
            }
        }
    }

    readCurrentPosition();

}

bool Scx11::prepareForMotorScan(Experiment &exp)
{
    Q_UNUSED(exp)
    return true;
}

void Scx11::checkLimitOneAxis(BlackChirp::MotorAxis axis)
{
    AxisInfo ai = axisInfo(axis);

    bool sigPositive, sigNegative;

    QByteArray resp = p_comm->queryCmd(QString("@%1@%1\n").arg(ai.id));
    if(resp.isEmpty())
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not read limits for %1 axis").arg(ai.name),BlackChirp::LogError);
        return;
    }

    resp = p_comm->queryCmd(QString("SIGLSP\n"));
    if(resp.contains("=0"))
    {
        sigPositive = false;
    }
    else if(resp.contains("=1"))
    {
        sigPositive = true;
    }
    else
    {
        emit hardwareFailure();
        emit logMessage(QString("Unable to check positive limit position for %1 axis.").arg(ai.name),BlackChirp::LogError);
        return;
    }

    resp = p_comm->queryCmd(QString("SIGLSN\n"));
    if(resp.contains("=0"))
    {
        sigNegative = false;
    }
    else if(resp.contains("=1"))
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

BlackChirp::MotorAxis Scx11::axisIndex(int id)
{
    switch(id)
    {
    case 0:
        return BlackChirp::MotorX;
        break;
    case 1:
        return BlackChirp::MotorY;
        break;
    case 2:
        return BlackChirp::MotorZ;
        break;
    default:
        return BlackChirp::MotorZ;
        break;
    }
}

Scx11::AxisInfo Scx11::axisInfo(BlackChirp::MotorAxis axis)
{
    switch(axis)
    {
    case BlackChirp::MotorX:
        return(d_channels.at(0));
        break;
    case BlackChirp::MotorY:
        return(d_channels.at(1));
        break;
    case BlackChirp::MotorZ:
        return(d_channels.at(2));
        break;
    default:
        return AxisInfo();
        break;
    }
}

QString Scx11::axisName(BlackChirp::MotorAxis axis)
{
    switch(axis)
    {
    case BlackChirp::MotorX:
        return QString("X");
        break;
    case BlackChirp::MotorY:
        return QString("Y");
        break;
    case BlackChirp::MotorZ:
        return QString("Z");
        break;
    default:
        return QString("T");
        break;
    }

    return QString();
}

bool Scx11::moving()
{
    for(int i=0; i<d_channels.size(); i++)
    {
        if(d_channels.at(i).moving)
            return true;
    }

    return false;
}
