#ifndef ZOOMPANPLOT_H
#define ZOOMPANPLOT_H

#include <qwt6/qwt_plot.h>
#include <QPoint>
#include <QEvent>
#include <QMenu>

class ZoomPanPlot : public QwtPlot
{
    Q_OBJECT

public:
    explicit ZoomPanPlot(QString name, QWidget *parent = nullptr);
    ~ZoomPanPlot();

    bool isAutoScale();
    void resetPlot();
    void setAxisAutoScaleRange(QwtPlot::Axis axis, double min, double max);
    void setAxisAutoScaleMin(QwtPlot::Axis axis, double min);
    void setAxisAutoScaleMax(QwtPlot::Axis axis, double max);
    void expandAutoScaleRange(QwtPlot::Axis axis, double newValueMin, double newValueMax);
    void setXRanges(const QwtScaleDiv &bottom, const QwtScaleDiv &top);

public slots:
    void autoScale();
    virtual void replot();
    void setZoomFactor(QwtPlot::Axis a, double v);

signals:
    void panningStarted();
    void panningFinished();
    void plotRightClicked(QMouseEvent *ev);

protected:
    const QString d_name;
    struct AxisConfig {
        QwtPlot::Axis type;
        bool autoScale;
        double min;
        double max;
        double zoomFactor;
        QString name;

        explicit AxisConfig(QwtPlot::Axis t, QString n) : type(t), autoScale(true),
            min(0.0), max(1.0), zoomFactor(0.1), name(n) {}
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

    virtual QMenu *contextMenu();

private:
    PlotConfig d_config;
    int getAxisIndex(QwtPlot::Axis a);
};

Q_DECLARE_METATYPE(QwtPlot::Axis)

#endif // ZOOMPANPLOT_H
