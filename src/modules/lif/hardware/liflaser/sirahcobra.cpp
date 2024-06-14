#include "sirahcobra.h"

using namespace BC::Key::LifLaser;

SirahCobra::SirahCobra(QObject *parent)
    : LifLaser{sCobra,sCobraName,CommunicationProtocol::Rs232,parent,true,true}
{
    setDefault(units,QString("nm"));
    setDefault(decimals,4);
    setDefault(minPos,450.0);
    setDefault(maxPos,700.0);

    if(!containsArray(stages))
    {
        appendArrayMap(stages,
                       {{sStart,3000.0},
                        {sHigh,12000.0},
                        {sRamp,2400},
                        {sMax,3300000},
                        {sbls,24000},
                        {sDat,""},
                       });
    }

}


void SirahCobra::initialize()
{
    p_comm->setReadOptions(200,false,"");
}

bool SirahCobra::testConnection()
{
    //work in progress
    QByteArray cmd;
    cmd.resize(13);
    cmd[0] = 0x3c;
    cmd[12] = 0x3e;
    cmd[1] = 0x17;
    for(int i=0; i<11; i++)
        cmd[11] = cmd.at(11) + cmd.at(i);

    p_comm->writeBinary(cmd);
    auto resp = p_comm->_device()->read(13);

    return true;
}

double SirahCobra::readPos()
{
}

void SirahCobra::setPos(double pos)
{
}

bool SirahCobra::readFl()
{
}

bool SirahCobra::setFl(bool en)
{
}


void SirahCobra::readSettings()
{
}
