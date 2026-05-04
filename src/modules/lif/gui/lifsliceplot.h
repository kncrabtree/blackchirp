#ifndef LIFSLICEPLOT_H
#define LIFSLICEPLOT_H

#include <gui/plot/zoompanplot.h>
#include <memory>

class QwtPlotCurve;
class QwtPlotTextLabel;

namespace BC::Key {
static const QString lifSliceCurve{"LifSliceCurve"};
}

/// \todo This needs to take an argument to differentiate time vs freq slice plots
class LifSlicePlot : public ZoomPanPlot
{
    Q_OBJECT
public:
    LifSlicePlot(const QString name, QWidget *parent = nullptr);
    ~LifSlicePlot();

    void prepareForExperiment();
    void setData(const QVector<QPointF> d, QString txt = "");

protected:
    std::unique_ptr<BlackchirpPlotCurve> p_curve;
    std::unique_ptr<QwtPlotTextLabel> p_label;
};

#endif // LIFSLICEPLOT_H
