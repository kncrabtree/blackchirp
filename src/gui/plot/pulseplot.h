#ifndef PULSEPLOT_H
#define PULSEPLOT_H

#include <src/gui/plot/zoompanplot.h>

#include <QList>

#include <src/data/experiment/pulsegenconfig.h>

class QwtPlotCurve;
class QwtPlotMarker;

namespace BC::Key {
static const QString pulsePlot("pulsePlot");
}

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
