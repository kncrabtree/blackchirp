#include <modules/motor/hardware/motorcontroller/motorcontroller.h>

#include <QTimer>

using namespace BC::Key::MC;

MotorController::MotorController(const QString subKey, const QString name, CommunicationProtocol::CommType commType, QObject *parent, bool threaded, bool critical) : HardwareObject(key,subKey,name,commType,parent,threaded,critical), d_idle(true)
{
    setDefault(xIndex,0);
    setDefault(yIndex,1);
    setDefault(zIndex,2);

    if(!containsArray(channels))
    {
        std::vector<SettingsMap> l;
        l.reserve(3);
        l.push_back({
                        {id,1},
                        {offset,100.0},
                        {min,-100.0},
                        {max,100.0},
                        {rest,-100.0},
                        {type,MotorScan::MotorX},
                        {axName,QString("X")},
                        {units,QString("mm")},
                        {decimal,1}
                    });
        l.push_back({
                        {id,2},
                        {offset,100.0},
                        {min,-100.0},
                        {max,100.0},
                        {rest,-100.0},
                        {type,MotorScan::MotorY},
                        {axName,QString("Y")},
                        {units,QString("mm")},
                        {decimal,1}
                    });
        l.push_back({
                        {id,3},
                        {offset,100.0},
                        {min,-100.0},
                        {max,100.0},
                        {rest,-100.0},
                        {type,MotorScan::MotorZ},
                        {axName,QString("Z")},
                        {units,QString("mm")},
                        {decimal,1}
                    });
        setArray(channels,l);
    }

    d_moving[MotorScan::MotorX] = false;
    d_moving[MotorScan::MotorY] = false;
    d_moving[MotorScan::MotorZ] = false;
}

bool MotorController::moveToPosition(double x, double y, double z)
{
    x = qBound(getArrayValue(channels,get(xIndex,0),min,-100.0),x,
               getArrayValue(channels,get(xIndex,0),max,100.0));
    y = qBound(getArrayValue(channels,get(yIndex,1),min,-100.0),y,
               getArrayValue(channels,get(yIndex,1),max,100.0));
    z = qBound(getArrayValue(channels,get(zIndex,2),min,-100.0),z,
               getArrayValue(channels,get(zIndex,2),max,100.0));

    if(p_motionTimer->isActive())
        p_motionTimer->stop();

    for(auto it = d_moving.begin(); it != d_moving.end(); ++it)
    {
        if(it.value())
        {
            if(!hwStopMotion(it.key()))
                return false;
            it.value() = false;
        }
    }

    if(hwMoveToPosition(x,y,z))
    {
        p_motionTimer->start();
        return true;
    }

    return false;
}

bool MotorController::prepareForExperiment(Experiment &exp)
{
    d_enabledForExperiment = exp.motorScan().isEnabled();
    if(d_enabledForExperiment)
        return prepareForMotorScan(exp);
    return true;
}

void MotorController::moveToRestingPos()
{
    moveToPosition(getArrayValue<double>(channels,get(xIndex,0),rest),
                   getArrayValue<double>(channels,get(yIndex,1),rest),
                   getArrayValue<double>(channels,get(zIndex,2),rest));
}

void MotorController::readCurrentPosition()
{
    for(auto ax : d_moving.keys())
    {
        auto p = hwReadPosition(ax);
        if(!isnan(p))
            emit posUpdate(ax,p,QPrivateSignal());
        else
            break;
    }
}

void MotorController::checkMotion()
{
    if(d_idle)
        return;

    bool moving = false;
    for(auto it = d_moving.begin(); it != d_moving.end(); ++it)
    {
        if(it.value())
        {
            it.value() = hwCheckAxisMotion(it.key());
            moving |= it.value();
        }
    }

    if(!moving)
    {
        d_idle = true;
        p_motionTimer->stop();
        emit motionComplete(true,QPrivateSignal());
    }
}

void MotorController::checkLimits()
{
    for(auto ax : d_moving.keys())
    {
        auto l = hwCheckLimits(ax);
        emit limitStatus(ax,l.first,l.second,QPrivateSignal());
    }
}

AxisInfo MotorController::getAxisInfo(MotorScan::MotorAxis a) const
{
    int id = -1;
    QString name{""};
    switch(a)
    {
    case MotorScan::MotorX:
        id = getArrayValue(channels,get(xIndex,0),::id,1);
        name = getArrayValue(channels,get(xIndex,0),::axName,QString("X"));
        break;
    case MotorScan::MotorY:
        id = getArrayValue(channels,get(yIndex,1),::id,2);
        name = getArrayValue(channels,get(yIndex,1),::axName,QString("Y"));
        break;
    case MotorScan::MotorZ:
        id = getArrayValue(channels,get(zIndex,2),::id,3);
        name = getArrayValue(channels,get(zIndex,2),::axName,QString("Z"));
        break;
    default:
        break;
    }

    return {id,name};
}

void MotorController::initialize()
{
    p_limitTimer = new QTimer(this);
    p_limitTimer->setInterval(get(lInterval,100));
    connect(p_limitTimer,&QTimer::timeout,this,&MotorController::checkLimits);
    connect(this,&MotorController::hardwareFailure,p_limitTimer,&QTimer::stop);

    p_motionTimer = new QTimer(this);
    p_motionTimer->setInterval(get(mInterval,50));
    connect(p_motionTimer,&QTimer::timeout,this,&MotorController::checkMotion);
    connect(this,&MotorController::hardwareFailure,p_motionTimer,&QTimer::stop);
    connect(this,&MotorController::hardwareFailure,[this](){
        emit motionComplete(false,QPrivateSignal());
    });

    mcInitialize();
}

bool MotorController::testConnection()
{
    if(mcTestConnection())
    {
        readCurrentPosition();
        checkLimits();
        p_limitTimer->start();
        return true;
    }

    return false;
}
