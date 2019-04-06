#ifndef VIRTUALGPIBCONTROLLER_H
#define VIRTUALGPIBCONTROLLER_H

#include "gpibcontroller.h"

class VirtualGpibController : public GpibController
{
	Q_OBJECT
public:
	VirtualGpibController(QObject *parent = 0);
	~VirtualGpibController();

	// HardwareObject interface
public slots:
    Experiment prepareForExperiment(Experiment exp);
	void beginAcquisition();
	void endAcquisition();

	// GpibController interface
protected:
    bool testConnection();
    void initialize();
    bool readAddress();
    bool setAddress(int a);

};

#endif // VIRTUALGPIBCONTROLLER_H
