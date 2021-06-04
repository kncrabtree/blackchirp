#ifndef LABJACKU3_H
#define LABJACKU3_H

#include <src/hardware/core/ioboard/ioboard.h>

#include "u3.h"

class LabjackU3 : public IOBoard
{
    Q_OBJECT
public:
    explicit LabjackU3(QObject *parent = nullptr);

    // HardwareObject interface
public slots:
    void readIOBSettings() override;
    Experiment prepareForExperiment(Experiment exp) override;

private:
    HANDLE d_handle;
    u3CalibrationInfo d_calInfo;

    int d_serialNo;

    bool configure();
    void closeConnection();
    QList<QPair<QString, QVariant> > auxData(bool plot);

    // HardwareObject interface
protected:
    bool testConnection() override;
    void initialize() override;
    virtual QList<QPair<QString, QVariant> > readAuxPlotData() override;
    virtual QList<QPair<QString, QVariant> > readAuxNoPlotData() override;
};

#endif // LABJACKU3_H
