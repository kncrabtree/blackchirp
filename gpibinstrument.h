#ifndef GPIBINSTRUMENT_H
#define GPIBINSTRUMENT_H

#include "communicationprotocol.h"
#include "gpibcontroller.h"

class GpibInstrument : public CommunicationProtocol
{
	Q_OBJECT
public:
	explicit GpibInstrument(QString key, QString subKey, GpibController *c, QObject *parent = nullptr);
	~GpibInstrument();
	void setAddress(int a);
	int address() const;

	QIODevice *device() { return nullptr; }

protected:
	GpibController *p_controller;
	int d_address;

	// CommunicationProtocol interface
public:
	bool writeCmd(QString cmd);
	QByteArray queryCmd(QString cmd);

public slots:
	void initialize();
	bool testConnection();
};

#endif // GPIBINSTRUMENT_H
