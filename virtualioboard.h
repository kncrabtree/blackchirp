#ifndef VIRTUALIOBOARD_H
#define VIRTUALIOBOARD_H

#include "ioboard.h"

class VirtualIOBoard : public IOBoard
{
    Q_OBJECT
public:
    explicit VirtualIOBoard(QObject *parent = nullptr);

    // HardwareObject interface
public slots:
    Experiment prepareForExperiment(Experiment exp);
    void beginAcquisition();
    void endAcquisition();

    // HardwareObject interface
protected:
    bool testConnection();
    void initialize();
    virtual QList<QPair<QString, QVariant> > readAuxPlotData();
    virtual QList<QPair<QString, QVariant> > readAuxNoPlotData();

private:
    QList<QPair<QString, QVariant> > auxData(bool plot);
};

#endif // VIRTUALIOBOARD_H
