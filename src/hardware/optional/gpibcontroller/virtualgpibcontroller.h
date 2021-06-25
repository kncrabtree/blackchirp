#ifndef VIRTUALGPIBCONTROLLER_H
#define VIRTUALGPIBCONTROLLER_H

#include <hardware/optional/gpibcontroller/gpibcontroller.h>

namespace BC::Key {
static const QString vgpibName("Virtual GPIB Controller");
}

class VirtualGpibController : public GpibController
{
	Q_OBJECT
public:
	VirtualGpibController(QObject *parent = 0);
	~VirtualGpibController();

protected:
    bool testConnection() override;
    void initialize() override;

    // GpibController interface
    bool readAddress() override;
    bool setAddress(int a) override;

};

#endif // VIRTUALGPIBCONTROLLER_H
