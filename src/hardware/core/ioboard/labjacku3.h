#ifndef LABJACKU3_H
#define LABJACKU3_H

#include <hardware/core/ioboard/ioboard.h>

#include "u3.h"

namespace BC::Key::IOB {
static const QString labjacku3("labjacku3");
static const QString labjacku3Name("Labjack U3 IO Board");
}

class LabjackU3 : public IOBoard
{
    Q_OBJECT
public:
    explicit LabjackU3(QObject *parent = nullptr);

    // HardwareObject interface
public slots:
    void readIOBSettings() override;
    bool prepareForExperiment(Experiment &exp) override;

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
