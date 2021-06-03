#include "gpibinstrument.h"

GpibInstrument::GpibInstrument(QString key, QString subKey, GpibController *c, QObject *parent) :
	CommunicationProtocol(CommunicationProtocol::Gpib,key,subKey,parent), p_controller(c)
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
	QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
	d_address = s.value(QString("%1/address").arg(key()),1).toInt();

	return true;
}
