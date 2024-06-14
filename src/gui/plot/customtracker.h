#ifndef CUSTOMTRACKER_H
#define CUSTOMTRACKER_H

#include <QPalette>
#include <QHash>
#include <QMap>

#include <qwt6/qwt_plot_canvas.h>
#include <qwt6/qwt_plot_picker.h>
#include <qwt6/qwt_plot.h>

class CustomTracker : public QwtPlotPicker
{
    Q_OBJECT
public:
    CustomTracker(QWidget *canvas);

    virtual QwtText trackerText( const QPoint &pos ) const;

    int axisDecimals(QwtPlot::Axis a) const { return d_details.value(a).decimals; }
    bool axisScientific(QwtPlot::Axis a) const { return d_details.value(a).scientific; }
    void setHorizontalTimeAxis(bool b) { d_hTime = b; }

public slots:
    void setDecimals(QwtPlot::Axis axis, int decimals);
    void setScientific(QwtPlot::Axis axis, bool scientific);

private:
    struct AxisDetails {
        int decimals{4};
        bool scientific{false};
    };
    QHash<QwtPlot::Axis,AxisDetails> d_details;
    QMap<QwtPlot::Axis,QString> d_axes{{QwtPlot::xBottom,QString("B")},
                                       {QwtPlot::xTop,QString("T")},
                                       {QwtPlot::yLeft,QString("L")},
                                       {QwtPlot::yRight,QString("R")}};

    bool d_hTime{false};
};

#endif // CUSTOMTRACKER_H
