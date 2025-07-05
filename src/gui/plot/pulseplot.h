#ifndef PULSEPLOT_H
#define PULSEPLOT_H

#include <gui/plot/zoompanplot.h>

#include <QList>
#include <memory>

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
    PulsePlot(std::shared_ptr<PulseGenConfig> cfg, QWidget *parent = 0);
    ~PulsePlot();

    struct PlotItem {
        double min;
        double max;
        double mid;
        std::unique_ptr<BlackchirpPlotCurve> curve;
        std::unique_ptr<QwtPlotMarker> labelMarker;
        std::unique_ptr<QwtPlotCurve> syncCurve;
        std::unique_ptr<QwtPlotMarker> separator;
        std::unique_ptr<QwtPlotMarker> separator2;
        
        // Make movable but not copyable
        PlotItem() = default;
        PlotItem(const PlotItem&) = delete;
        PlotItem& operator=(const PlotItem&) = delete;
        PlotItem(PlotItem&&) = default;
        PlotItem& operator=(PlotItem&&) = default;
    };

public slots:
    void updatePulsePlot();
    void newConfig(std::shared_ptr<PulseGenConfig> c);

    // ZoomPanPlot interface
protected:
    void replot() override;

private:
    std::weak_ptr<PulseGenConfig> ps_config;
    std::vector<PlotItem> d_plotItems;


    // QWidget interface
public:
    QSize sizeHint() const override;
};

#endif // PULSEPLOT_H
