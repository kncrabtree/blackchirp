#ifndef ZOOMPANPLOT_H
#define ZOOMPANPLOT_H

#include <qwt6/qwt_plot.h>
#include <QPoint>
#include <QEvent>

class ZoomPanPlot : public QwtPlot
{
    Q_OBJECT

public:
    explicit ZoomPanPlot(QWidget *parent = nullptr);
    ~ZoomPanPlot();

    bool isAutoScale();
    void resetPlot();
    void autoScale();
    void setAxisAutoScaleRange(QwtPlot::Axis axis, double min, double max);
    void setAxisAutoScaleMin(QwtPlot::Axis axis, double min);
    void setAxisAutoScaleMax(QwtPlot::Axis axis, double max);
    void expandAutoScaleRange(QwtPlot::Axis axis, double newValueMin, double newValueMax);

public slots:
    virtual void replot();

signals:
    void panningStarted();
    void panningFinished();
    void plotRightClicked(QMouseEvent *ev);

protected:
    struct AxisConfig {
        QwtPlot::Axis type;
        bool autoScale;
        double min;
        double max;
        double zoomFactor;

        explicit AxisConfig(QwtPlot::Axis t) : type(t), autoScale(true), min(0.0), max(1.0), zoomFactor(0.1) {}
    };

    struct PlotConfig {
        QList<AxisConfig> axisList;

        bool xDirty;
        bool panning;
        QPoint panClickPos;

        PlotConfig() : xDirty(false), panning(false) {}
    };

    virtual void filterData() =0;
    virtual void resizeEvent(QResizeEvent *ev);
    virtual bool eventFilter(QObject *obj, QEvent *ev);
    virtual void pan(QMouseEvent *me);
    virtual void zoom(QWheelEvent *we);

private:
    PlotConfig d_config;
    int getAxisIndex(QwtPlot::Axis a);
};

Q_DECLARE_METATYPE(QwtPlot::Axis)

#endif // ZOOMPANPLOT_H
