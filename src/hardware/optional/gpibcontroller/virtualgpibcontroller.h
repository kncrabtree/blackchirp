#ifndef VIRTUALGPIBCONTROLLER_H
#define VIRTUALGPIBCONTROLLER_H

#include "gpibcontroller.h"

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
