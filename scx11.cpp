#include "scx11.h"

Scx11::Scx11(QObject *parent) : MotorController(parent)
{

}



bool Scx11::testConnection()
{
    if(!p_comm->testConnection())
    {

    }
}

void Scx11::initialize()
{
    p_comm->initialize();
    p_comm->setReadOptions(1000,true,QByteArray("\r\n"));
    testConnection();
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
}

bool Scx11::prepareForMotorScan(const MotorScan ms)
{
}

void Scx11::moveToRestingPos()
{
}

void Scx11::checkLimit()
{
}
