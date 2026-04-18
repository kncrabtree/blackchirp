#ifndef VIRTUALGPIBCONTROLLER_H
#define VIRTUALGPIBCONTROLLER_H

#include <hardware/optional/gpibcontroller/gpibcontroller.h>
#include <QThread>
#include <QDebug>

class VirtualGpibController : public GpibController
{
	Q_OBJECT
public:
	VirtualGpibController(const QString& label, QObject *parent = nullptr);
	~VirtualGpibController();

    // Override communication methods for debug output and multi-threading testing
    bool writeCmd(int address, const QString &cmd);
    bool writeBinary(int address, const QByteArray &dat);
    QByteArray queryCmd(int address, const QString &cmd, bool suppressError=false);

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
