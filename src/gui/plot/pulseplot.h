#ifndef PULSEPLOT_H
#define PULSEPLOT_H

#include <gui/plot/zoompanplot.h>

#include <QList>

#include <hardware/optional/pulsegenerator/pulsegenconfig.h>

class QwtPlotCurve;
class QwtPlotMarker;

namespace BC::Key {
static const QString pulsePlot{"pulsePlot"};
static const QString pulseChannel{"PulseChannel"};
}

class PulsePlot : public ZoomPanPlot
{
    Q_OBJECT
public:
    PulsePlot(QWidget *parent = 0);
    ~PulsePlot();

    struct PlotItem {
        double min;
        double max;
        double mid;
        BlackchirpPlotCurve *curve;
        QwtPlotMarker *labelMarker;
        QwtPlotCurve *syncCurve;
    };

public slots:
    void newConfig(const PulseGenConfig &c);

    // ZoomPanPlot interface
protected:
    void replot() override;

private:
    PulseGenConfig d_config;
    QVector<PlotItem> d_plotItems;


    // QWidget interface
public:
    QSize sizeHint() const override;
};

#endif // PULSEPLOT_H
