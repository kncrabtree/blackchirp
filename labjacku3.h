#ifndef LABJACKU3_H
#define LABJACKU3_H

#include "ioboard.h"

#include "u3.h"

class LabjackU3 : public IOBoard
{
    Q_OBJECT
public:
    explicit LabjackU3(QObject *parent = nullptr);

    // HardwareObject interface
public slots:
    void readSettings();
    Experiment prepareForExperiment(Experiment exp);
    void beginAcquisition();
    void endAcquisition();
    void readTimeData();

private:
    HANDLE d_handle;
    u3CalibrationInfo d_calInfo;

    int d_serialNo;

    bool configure();
    void closeConnection();
    QList<QPair<QString, QVariant> > auxData(bool plot);

    // HardwareObject interface
protected:
    bool testConnection();
    void initialize();
    virtual QList<QPair<QString, QVariant> > readAuxPlotData();
    virtual QList<QPair<QString, QVariant> > readAuxNoPlotData();
};

#endif // LABJACKU3_H
