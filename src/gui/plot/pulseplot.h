#ifndef PULSEPLOT_H
#define PULSEPLOT_H

#include <src/gui/plot/zoompanplot.h>

#include <QList>

#include <src/data/experiment/pulsegenconfig.h>

class QwtPlotCurve;
class QwtPlotMarker;

namespace BC::Key {
static const QString pulsePlot("pulsePlot");
static const QString pulseChannel("PulseChannel");
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
    void newSetting(int index, PulseGenConfig::Setting s, QVariant val);
    void newRepRate(double d);

    // ZoomPanPlot interface
protected:
    void replot();

private:
    PulseGenConfig d_config;
    QList<QPair<BlackchirpPlotCurve*,QwtPlotMarker*>> d_plotItems;

};

#endif // PULSEPLOT_H
