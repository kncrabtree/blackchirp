#include "gpibinstrument.h"

GpibInstrument::GpibInstrument(QString key, GpibController *c, QObject *parent) :
    CommunicationProtocol(key,parent), p_controller(c)
{
}

GpibInstrument::~GpibInstrument()
{

}

void GpibInstrument::setAddress(int a)
{
	d_address = a;
}

int GpibInstrument::address() const
{
	return d_address;
}



bool GpibInstrument::writeCmd(QString cmd)
{
    return p_controller->writeCmd(d_address,cmd);
}

bool GpibInstrument::writeBinary(QByteArray dat)
{
    return p_controller->writeBinary(d_address,dat);
}

QByteArray GpibInstrument::queryCmd(QString cmd, bool suppressError)
{
    return p_controller->queryCmd(d_address,cmd,suppressError);
}

void GpibInstrument::initialize()
{
}

bool GpibInstrument::testConnection()
{
    SettingsStorage s(d_key,SettingsStorage::Hardware);

    d_address = s.get<int>("address",1);

	return true;
}
