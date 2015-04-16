#ifndef PULSEPLOT_H
#define PULSEPLOT_H

#include "zoompanplot.h"
#include "pulsegenconfig.h"
#include <QList>
#include <qwt6/qwt_plot_curve.h>
#include <qwt6/qwt_plot_marker.h>

class PulsePlot : public ZoomPanPlot
{
    Q_OBJECT
public:
    PulsePlot(QWidget *parent = 0);
    ~PulsePlot();

public slots:
    void newConfig(const PulseGenConfig c);
    void newSetting(int index, PulseGenConfig::Setting s, QVariant val);

    // ZoomPanPlot interface
protected:
    void filterData();
    void replot();

private:
    PulseGenConfig d_config;
    QList<QPair<QwtPlotCurve*,QwtPlotMarker*>> d_plotItems;

};

#endif // PULSEPLOT_H
