#ifndef VIRTUALGPIBCONTROLLER_H
#define VIRTUALGPIBCONTROLLER_H

#include <hardware/optional/gpibcontroller/gpibcontroller.h>
#include <QThread>
#include <QDebug>

namespace BC::Key {
static const QString vgpibName("Virtual GPIB Controller");
}

class VirtualGpibController : public GpibController
{
	Q_OBJECT
public:
	VirtualGpibController(QObject *parent = 0);
	VirtualGpibController(const QString& subKey, QObject *parent = 0);
	~VirtualGpibController();

    // Override communication methods for debug output and multi-threading testing
    bool writeCmd(int address, QString cmd);
    bool writeBinary(int address, QByteArray dat);
    QByteArray queryCmd(int address, QString cmd, bool suppressError=false);

protected:
    bool testConnection() override;
    void initialize() override;

    // GpibController interface
    bool readAddress() override;
    bool setAddress(int a) override;

private:
    QString getThreadInfo() const;
};

#endif // VIRTUALGPIBCONTROLLER_H
