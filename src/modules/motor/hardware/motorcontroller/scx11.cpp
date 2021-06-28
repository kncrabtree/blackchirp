#include "scx11.h"

using namespace BC::Key::MC;

Scx11::Scx11(QObject *parent) :
    MotorController(scx11,scx11Name,CommunicationProtocol::Rs232,parent)
{
}

bool Scx11::mcTestConnection()
{
    QByteArray resp;

    auto l = getArray(channels);

    for(auto m : l)
    {
        int id = m.at(::id).toInt();
        QString name = m.at(axName).toString();

        resp = p_comm->queryCmd(QString("@%1@%1\n").arg(id));
        if(resp.isEmpty())
        {
            d_errorString = QString("Could not communicate with %1 axis").arg(name);
            return false;
        }

        resp = p_comm->queryCmd(QString("VER\n"));
        if(resp.isEmpty())
        {
            d_errorString = QString("Could not get version info from %1 axis").arg(name);
            return false;
        }

        if(resp.startsWith("VER"))
        {
            QByteArray t = p_comm->queryCmd(QString("ECHO=0\n"));
            if(t.isEmpty())
            {
                d_errorString = QString("Could not disable echo on %1 axis").arg(name);
                return false;
            }

            resp = p_comm->queryCmd(QString("VER\n"));
            if(resp.isEmpty())
            {
                d_errorString = QString("Could not get version info from %1 axis").arg(name);
                return false;
            }
        }

        if(!resp.startsWith("SCX11"))
        {
            d_errorString =  QString("Could not connect to SCX11. ID response: %1").arg(QString(resp.trimmed()));
            return false;
        }
        emit logMessage(QString("ID response: %1").arg(QString(resp.trimmed())));

        QByteArray t = p_comm->queryCmd(QString("VERBOSE=1\n"));
        if(t.isEmpty())
        {
            d_errorString = QString("Could not enable verbose mode on %1 axis").arg(name);
            return false;
        }


        //read negative limit
        resp = p_comm->queryCmd(QString("SIGLSN\n"));
        if(!resp.contains("=1"))
        {
            d_errorString = QString("%1 axis is not at its negative limit. Move it there manually and reconnect.").arg(name);
            return false;
        }


        t = p_comm->queryCmd(QString("PC=%1\n").arg(-m.at(offset).toDouble(),0,'f',3));
        if(t.isEmpty())
        {
            d_errorString = QString("Could not set initial home offset on %1 axis").arg(name);
            return false;
        }

    }

    //set speed of Z axis
    p_comm->writeCmd(QString("@3@3\n"));
    p_comm->writeCmd(QString("VR 10\n"));

    return true;
}

void Scx11::mcInitialize()
{
    p_comm->setReadOptions(1000,true,QByteArray(">"));
}

bool Scx11::hwMoveToPosition(double x, double y, double z)
{
    QList<QPair<int,double>> positions{
        {getArrayValue(channels,get(xIndex,0),id,1),x},
        {getArrayValue(channels,get(yIndex,1),id,2),y},
        {getArrayValue(channels,get(zIndex,2),id,3),z}, };

    for(auto p : positions)
    {
        QByteArray resp = p_comm->queryCmd(QString("@%1@%1\n").arg(p.first));
        resp = p_comm->queryCmd(QString("MA %1\n").arg(p.second,0,'f',3));
    }

    return true;
}

bool Scx11::prepareForMotorScan(Experiment &exp)
{
    Q_UNUSED(exp)
    return true;
}

Limits Scx11::hwCheckLimits(MotorScan::MotorAxis axis)
{

    auto [id,name] = getAxisInfo(axis);

    if(id < 0)
        return {false,false};

    QByteArray resp = p_comm->queryCmd(QString("@%1@%1\n").arg(id));
    if(resp.isEmpty())
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not read limits for %1 axis").arg(name),BlackChirp::LogError);
        return {false,false};
    }

    bool sigPositive, sigNegative;

    resp = p_comm->queryCmd(QString("SIGLSP\n"));
    if(resp.contains("=0"))
        sigPositive = false;
    else if(resp.contains("=1"))
        sigPositive = true;
    else
    {
        emit hardwareFailure();
        emit logMessage(QString("Unable to check positive limit position for %1 axis. Response: %2").arg(name).arg(QString(resp)),BlackChirp::LogError);
        return {false,false};
    }

    resp = p_comm->queryCmd(QString("SIGLSN\n"));
    if(resp.contains("=0"))
        sigNegative = false;
    else if(resp.contains("=1"))
        sigNegative = true;
    else
    {
        emit hardwareFailure();
        emit logMessage(QString("Unable to check negative limit position for %1 axis. Response: %2").arg(name).arg(QString(resp)),BlackChirp::LogError);
        return {false,false};
    }

    return {sigNegative,sigPositive};
}

double Scx11::hwReadPosition(MotorScan::MotorAxis axis)
{
    auto [id,name] = getAxisInfo(axis);

    if(id < 0)
        return nan("");

    QByteArray resp = p_comm->queryCmd(QString("@%1@%1\n").arg(id));
    if(resp.isEmpty())
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not read position for %1 axis").arg(name),BlackChirp::LogError);
        return false;
    }

    resp = p_comm->queryCmd(QString("PC\n"));
    if(resp.isEmpty() || !resp.contains('=') || !resp.contains('m'))
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not read position for %1 axis. Response: %2").arg(name).arg(QString(resp)),BlackChirp::LogError);
        return false;
    }
    int f = resp.indexOf('=');
    int l = resp.indexOf('m',f);
    return resp.mid(f+1,l-f-1).trimmed().toDouble();

}

bool Scx11::hwCheckAxisMotion(MotorScan::MotorAxis axis)
{
    auto [id,name] = getAxisInfo(axis);

    QByteArray resp = p_comm->queryCmd(QString("@%1@%1\n").arg(id));
    resp = p_comm->queryCmd(QString("SIGMOVE\n"));
    if(resp.isEmpty())
    {
        emit hardwareFailure();
        emit logMessage(QString("Error occured during motion of %1 axis. Sequence aborted.").arg(name),BlackChirp::LogError);
        return false;

    }
    if(resp.contains("=0"))
        return false;

    return true;

}

bool Scx11::hwStopMotion(MotorScan::MotorAxis axis)
{
    auto [id,name] = getAxisInfo(axis);

    QByteArray resp = p_comm->queryCmd(QString("@%1@%1\n").arg(id));
    resp = p_comm->queryCmd(QString("HSTOP\n"));
    if(resp.isEmpty())
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not abort current motion of %1 axis").arg(name),BlackChirp::LogError);
        return false;
    }

    return true;
}
