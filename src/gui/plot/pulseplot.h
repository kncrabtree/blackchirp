#ifndef PULSEPLOT_H
#define PULSEPLOT_H

#include "zoompanplot.h"

#include <QList>

#include "pulsegenconfig.h"

class QwtPlotCurve;
class QwtPlotMarker;

class PulsePlot : public ZoomPanPlot
{
    Q_OBJECT
public:
    PulsePlot(QWidget *parent = 0);
    ~PulsePlot();

    PulseGenConfig config();

public slots:
    void newConfig(const PulseGenConfig c);
    void newSetting(int index, BlackChirp::PulseSetting s, QVariant val);
    void newRepRate(double d);

    // ZoomPanPlot interface
protected:
    void filterData();
    void replot();

private:
    PulseGenConfig d_config;
    QList<QPair<QwtPlotCurve*,QwtPlotMarker*>> d_plotItems;

};

#endif // PULSEPLOT_H
