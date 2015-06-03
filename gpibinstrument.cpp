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

QByteArray GpibInstrument::queryCmd(QString cmd)
{
	return p_controller->queryCmd(d_address,cmd);
}

void GpibInstrument::initialize()
{
	testConnection();
}

bool GpibInstrument::testConnection()
{
	QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
	d_address = s.value(QString("%1/address").arg(key()),1).toInt();

	return true;
}
