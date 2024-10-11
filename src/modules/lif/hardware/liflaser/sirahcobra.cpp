#include "sirahcobra.h"

#include <QtEndian>
#include <math.h>
#include <QThread>

#ifndef M_PI
#define M_PI 3.1415926535897323846
#endif

using namespace BC::Key::LifLaser;

SirahCobra::SirahCobra(QObject *parent)
    : LifLaser{sCobra,sCobraName,CommunicationProtocol::Rs232,parent,true,true}
{
    setDefault(units,QString("nm"));
    setDefault(decimals,4);
    setDefault(minPos,450.0);
    setDefault(maxPos,700.0);
    setDefault(hasFl,false);
    setDefault(hasExtStage,true);
    setDefault(extStagePort,QString("COM9"));
    setDefault(extStageBaud,9600);
    setDefault(extStageCrystalAddress,8);
    setDefault(extStageCompAddress,2);
    setDefault(extStageCrystalTheta0,123.7593111);
    setDefault(extStageCrystalSlope,0.3219747226);
    setDefault(extStageCompTheta0,518.1670078);
    setDefault(extStageCompSlope,-0.3820040691);

    if(!containsArray(stages))
    {
        appendArrayMap(stages,
                       {{sStart,3000.0},
                        {sHigh,12000.0},
                        {sRamp,2400},
                        {sMax,3300000},
                        {sbls,24000},
                        {sLeverLength,134.599318},
                        {sLinearOffset,-76.543335},
                        {sAngleOffset,31.329809},
                        {sGrooves,2414.0},
                        {sGrazingAngle,85.0},
                        {sPitch,-0.25},
                        {sGrooves,2414.0},
                        {sMotorResolution,4800},
                       },true);
    }
    
    if(!containsArray(extStageCrystalPoly))
    {
        setArray(extStageCrystalPoly,{
                     {{polyOrder,0},{polyValue,3.95707878e10}},
                     {{polyOrder,1},{polyValue,-2.79608834e8}},
                     {{polyOrder,2},{polyValue,6.17881467e5}},
                     {{polyOrder,3},{polyValue,-3.48885374}},
                     {{polyOrder,4},{polyValue,-1.91691352}},
                     {{polyOrder,5},{polyValue,2.71068206e-3}},
                     {{polyOrder,6},{polyValue,-1.19684033e-6}}
                 });
    }
    
    if(!containsArray(extStageCompPoly))
    {
        setArray(extStageCompPoly,{
                     {{polyOrder,0},{polyValue,735434.26345}},
                     {{polyOrder,1},{polyValue,-2083.62338}},
                     {{polyOrder,2},{polyValue,1.80616}},
                 });
    }

    save();


}


void SirahCobra::initialize()
{
    p_comm->setReadOptions(200,false,"");

    if(get(hasExtStage,false))
    {
        p_extStagePort = new Rs232Instrument(d_key+"ExtStage",this);
        connect(p_extStagePort,&Rs232Instrument::logMessage,this,&SirahCobra::logMessage);
        connect(p_extStagePort,&Rs232Instrument::hardwareFailure,this,&SirahCobra::hardwareFailure);
        p_extStagePort->initialize();
    }
}

bool SirahCobra::testConnection()
{
    bool out = prompt();
    if(out)
    {
        //disable autoprompt
        p_comm->writeBinary(buildCommand(0x10));
        readPosition();
    }

    if(p_extStagePort)
    {
        p_extStagePort->setReadOptions(200,true,"\r\n");
        if(!p_extStagePort->testManual(get(extStagePort,QString("COM9"))
                                        ,get(extStageBaud,9600)))
        {
            d_errorString = "Could not open external stage port.";
            return false;
        }

        auto dev = QString::number(get(extStageCrystalAddress,0));
        auto resp = p_extStagePort->queryCmd(dev+"in");
        bool ok;
        double range = static_cast<double>(resp.mid(21,4).toInt(&ok,16));
        double pp = static_cast<double>(resp.mid(25,8).toInt(&ok,16));
        d_crystalStatus.stepsPerDeg = pp/range;
        dev = QString::number(get(extStageCompAddress,0));
        resp = p_extStagePort->queryCmd(dev+"in");
        range = static_cast<double>(resp.mid(21,4).toInt(&ok,16));
        pp = static_cast<double>(resp.mid(25,8).toInt(&ok,16));
        d_compStatus.stepsPerDeg = pp/range;
    }

    return out;
}

double SirahCobra::readPos()
{
    if(!prompt())
    {
        emit logMessage(QString("Could not read position."),LogHandler::Error);
        emit hardwareFailure();
        return 0.0;
    }

    return posToWavelength(d_status.m1Pos);
}

void SirahCobra::setPos(double pos)
{
    if(!prompt())
    {
        emit logMessage(QString("Could not set position to %1").arg(pos,0,'f',get(decimals,2)),LogHandler::Error);
        return;
    }

    //calculate target position
    auto targetPos = wavelengthToPos(pos);
    auto currentPos = d_status.m1Pos;
    auto delta = targetPos - currentPos;

    //if the calculated move is too small, just assume we're good enough
    if(qAbs(delta) < 10)
        return;

    //can we just move relative?
    //Conditions: need last move to be in same direction as backlash correction,
    //and distance should be less than backlash correction.
    auto backlash = getArrayValue(stages,0,sbls,-24000);
    if(d_status.lastMoveDir != 0 && (d_status.lastMoveDir*delta) > 0 && qAbs(delta) < qAbs(backlash))
    {
        moveRelative(delta);
        if(backlash > 0)
            d_status.lastMoveDir = 1;
        else
            d_status.lastMoveDir = -1;
    }
    else
    {
        if(moveAbsolute(targetPos - backlash))
        {
            moveRelative(backlash);
            if(backlash > 0)
                d_status.lastMoveDir = 1;
            else
                d_status.lastMoveDir = -1;
        }
        else
            d_status.lastMoveDir = 0;
    }

    if(p_extStagePort)
    {
        double cp = 0.0;
        for(const auto &[o,v] : d_crystalStatus.coefs)
            cp += pow(pos,o)*v;
        qint32 crystalPos = static_cast<qint32>(round(cp));
        
        cp = 0.0;
        for(const auto &[o,v] : d_compStatus.coefs)
            cp += pow(pos,o)*v;
        qint32 compPos = static_cast<qint32>(round(cp));

        // emit logMessage("Crystal: "+QString::number(crystalPos));
        // emit logMessage("Compensator: "+QString::number(compPos));
        
        //calculate crystal commands
        auto dev = QString::number(get(extStageCrystalAddress,0));
        QString absmove = QString::number(crystalPos,16).rightJustified(8,'0').toUpper();
        QString crysCmd1 = QString("%1ma%2").arg(dev,absmove);
        // emit logMessage(QString("Crystal command: "+crysCmd1));

        dev = QString::number(get(extStageCompAddress,0));
        absmove = QString::number(compPos,16).rightJustified(8,'0').toUpper();
        QString compCmd1 = QString("%1ma%2").arg(dev,absmove);
        // emit logMessage(QString("Compensator command: "+compCmd1));
        
        std::vector<QString> cmds {crysCmd1,compCmd1};
        for(const auto &c : cmds)
        {
           p_extStagePort->writeCmd(c);
           int count = 0;
           int ba = p_extStagePort->_device()->bytesAvailable();
           while(ba < 13)
           {
               count++;
               if(count > 25)
               {
                   emit hardwareFailure();
                   emit logMessage(QString("Error in command %1. No response received.").arg(c),LogHandler::Error);
                   return;
               }
               p_extStagePort->_device()->waitForReadyRead(250);
               ba = p_extStagePort->_device()->bytesAvailable();
           }
        
           auto resp = p_extStagePort->_device()->readAll();
        }
    }
}

bool SirahCobra::readFl()
{
    return true;
}

bool SirahCobra::setFl(bool en)
{
    Q_UNUSED(en)
    return true;
}


void SirahCobra::readSettings()
{
    d_params.clear();
    for(uint i=0; i<getArraySize(stages); i++)
    {
        TuningParameters tp;
        tp.angOff = getArrayValue(stages,i,sAngleOffset).toDouble()/180*M_PI;
        tp.grazAng = getArrayValue(stages,i,sGrazingAngle).toDouble()/180*M_PI;
        tp.grooves = getArrayValue(stages,i,sGrooves).toDouble();
        tp.lLen = getArrayValue(stages,i,sLeverLength).toDouble();
        tp.linOff = getArrayValue(stages,i,sLinearOffset).toDouble();
        tp.mRes = getArrayValue(stages,i,sMotorResolution).toDouble();
        tp.pitch = getArrayValue(stages,i,sPitch).toDouble();
        d_params.push_back(tp);
    }

    if(p_extStagePort)
    {        
        d_crystalStatus.coefs.clear();
        auto l = getArray(extStageCrystalPoly);
        if(l.empty())
        {
            d_crystalStatus.coefs.insert({0.0,0.0});
            d_crystalStatus.coefs.insert({1.0,1.0});
        }
        else
        {
            for(const auto &m : l)
            {
                double order = 0.0;
                if(m.contains(polyOrder))
                    order = m.at(polyOrder).toDouble();
                
                double val = 0.0;
                if(m.contains(polyValue))
                    val = m.at(polyValue).toDouble();
                
                d_crystalStatus.coefs.insert({order,val});
            }
        }
        
        d_compStatus.coefs.clear();
        l = getArray(extStageCompPoly);
        if(l.empty())
        {
            d_compStatus.coefs.insert({0.0,0.0});
            d_compStatus.coefs.insert({1.0,1.0});
        }
        else
        {
            for(const auto &m : l)
            {
                double order = 0.0;
                if(m.contains(polyOrder))
                    order = m.at(polyOrder).toDouble();
                
                double val = 0.0;
                if(m.contains(polyValue))
                    val = m.at(polyValue).toDouble();
                
                d_compStatus.coefs.insert({order,val});
            }
        }
    }
}

QByteArray SirahCobra::buildCommand(char cmd, QByteArray args)
{
    //work in progress
    QByteArray out;
    out.fill(0x00,13);
    out[0] = 0x3c;
    out[1] = cmd;
    out[11] += out[0];
    out[11] += out[1];
    for(int i=0; i<args.size() && i<9; i++)
    {
        out[i+2] = args.at(i);
        out[11] += args.at(i);
    }
    out[12] = 0x3e;

    return out;
}

bool SirahCobra::prompt()
{
    auto rp = buildCommand(0x17);
    p_comm->writeBinary(rp);
    auto resp = p_comm->readBytes(14,true);

    if(resp.size() != 14 || !resp.startsWith(0x5b) || !resp.endsWith(0x5d))
    {
        d_errorString = QString("Received unexpected response (Hex: %1)").arg(QString(resp.toHex()));
        return false;
    }

    d_status.err = static_cast<quint8>(resp.at(1));
    d_status.cStatus = static_cast<quint8>(resp.at(2));
    d_status.m1Status = static_cast<quint8>(resp.at(3));
    qint32 pos = 0;
    pos |= static_cast<quint8>(resp.at(4));
    pos |= (static_cast<quint8>(resp.at(5)) << 8);
    pos |= (static_cast<quint8>(resp.at(6)) << 16);
    pos |= (static_cast<quint8>(resp.at(7)) << 24);
    d_status.m1Pos = pos;
    d_status.m2Status = static_cast<quint8>(resp.at(8));
    pos = 0;
    pos |= static_cast<quint8>(resp.at(9));
    pos |= (static_cast<quint8>(resp.at(10)) << 8);
    pos |= (static_cast<quint8>(resp.at(11)) << 16);
    pos |= (static_cast<quint8>(resp.at(12)) << 24);
    d_status.m2Status = pos;

    return true;

}

double SirahCobra::posToWavelength(qint32 pos, uint stage)
{
    if(stage >= d_params.size())
        return 0.0;

    const auto &tp = d_params.at(stage);
    auto x = tp.linOff - (tp.pitch/tp.mRes)*static_cast<double>(pos);
    auto phi_o = tp.angOff - asin(x/tp.lLen);
    auto wl = (sin(tp.grazAng) + sin(phi_o))/tp.grooves;

    return wl *1e6;
}

qint32 SirahCobra::wavelengthToPos(double wl, uint stage)
{
    if(stage >= d_params.size())
        return 0;

    const auto &tp = d_params.at(stage);
    auto phi_o = asin(tp.grooves*wl/1e6 - sin(tp.grazAng));
    auto x = tp.linOff - tp.lLen*sin(tp.angOff - phi_o);
    auto p = tp.mRes/tp.pitch*x;

    return static_cast<qint32>(round(p));
}

void SirahCobra::moveRelative(qint32 steps)
{
    quint8 dir = 0x01;
    if(steps < 0)
        dir = 0x02;

    auto s = qAbs(steps);
    QByteArray dat;
    dat.append(0x01);
    dat.append(dir);
    dat.append(static_cast<quint8>(s & 0x000000ff));
    dat.append(static_cast<quint8>((s & 0x0000ff00) >> 8));
    dat.append(static_cast<quint8>((s & 0x00ff0000) >> 16));
    dat.append(static_cast<quint8>((s & 0xff000000) >> 24));

    auto cmd = buildCommand(0x06,dat);

    p_comm->writeBinary(cmd);

    int waiting = 0;
    bool done = false;

    while(!done && waiting < 100)
    {
        thread()->msleep(50);

        if(!prompt())
            break;

        // emit logMessage(QString("Target (rel): %1, Current: %2").arg(steps).arg(d_status.m1Pos));

        //bit 0 tells whether the motor is running
        if(d_status.m1Status % 2)
        {
            //motor is running; sleep thread and try again
            waiting++;
        }
        else
        {
            done = true;
            break;
        }
    }

    if(!done)
    {
        //stop motor
        p_comm->writeBinary(buildCommand(0x04));
        emit logMessage(QString("Did not set position successfully; stopped motor motion."),LogHandler::Error);
        emit hardwareFailure();
    }

}

bool SirahCobra::moveAbsolute(qint32 targetPos)
{
    QByteArray dat;
    dat.append(0x01);
    dat.append(static_cast<quint8>(targetPos & 0x000000ff));
    dat.append(static_cast<quint8>((targetPos & 0x0000ff00) >> 8));
    dat.append(static_cast<quint8>((targetPos & 0x00ff0000) >> 16));
    dat.append(static_cast<quint8>((targetPos & 0xff000000) >> 24));

    auto cmd = buildCommand(0x07,dat);

    p_comm->writeBinary(cmd);

    int waiting = 0;
    bool done = false;
    qint32 lastDiff = qAbs(targetPos - d_status.m1Pos);

    while(!done)
    {
        thread()->msleep(50);

        if(!prompt())
            break;

        if(waiting > 0)
        {
            auto d = qAbs(targetPos - d_status.m1Pos);
            if(d > lastDiff && d > 10)
            {
                emit logMessage(QString("Diff increased!"),LogHandler::Error);
                break;
            }
            lastDiff = d;
        }
        // emit logMessage(QString("Target: %1, Current: %2").arg(targetPos).arg(d_status.m1Pos));

        //bit 0 tells whether the motor is running
        if(d_status.m1Status % 2)
        {
            //motor is running; sleep thread and try again
            waiting++;
        }
        else
        {
            done = true;
            break;
        }
    }

    if(!done)
    {
        //stop motor
        p_comm->writeBinary(buildCommand(0x04));
        emit logMessage(QString("Did not set position successfully; stopped motor motion."),LogHandler::Error);
        emit hardwareFailure();
    }

    return done;
}

