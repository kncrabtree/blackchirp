#ifndef GPIBINSTRUMENT_H
#define GPIBINSTRUMENT_H

#include <hardware/core/communication/communicationprotocol.h>
#include <data/settings/hardwarekeys.h>

class GpibController;


class GpibInstrument : public CommunicationProtocol
{
	Q_OBJECT
public:
    explicit GpibInstrument(QString key, GpibController *c, QObject *parent = nullptr);
	virtual ~GpibInstrument();
	void setAddress(int a);
	int address() const;

protected:
	GpibController *p_controller;
	int d_address;
	QString d_ownerKey;  // Store the parent's key to avoid pointer access during destruction

	// CommunicationProtocol interface
public:
    bool writeCmd(const QString &cmd) override;
    bool writeBinary(const QByteArray &dat) override;
    QByteArray queryCmd(const QString &cmd, bool suppressError=false) override;

public slots:
    void initialize() override;
    bool testConnection() override;

private slots:
    void onControllerDestroyed();
    
    // CommunicationProtocol interface
public:
    QIODevice *_device() override;
};

#endif // GPIBINSTRUMENT_H
