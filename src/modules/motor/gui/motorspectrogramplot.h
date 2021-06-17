#ifndef MOTORSPECTROGRAMPLOT_H
#define MOTORSPECTROGRAMPLOT_H

#include <src/gui/plot/zoompanplot.h>

#include <qwt6/qwt_matrix_raster_data.h>
#include <qwt6/qwt_plot_spectrogram.h>

#include <src/modules/motor/data/motorscan.h>

namespace BC::Key {
static const QString leftAxis("leftAxis");
static const QString bottomAxis("bottomAxis");
static const QString slider1Axis("slider1Axis");
static const QString slider2Axis("slider2Axis");
}

class MotorSpectrogramPlot : public ZoomPanPlot
{
    Q_OBJECT
public:
    MotorSpectrogramPlot(const QString name, QWidget *parent = nullptr);
    virtual ~MotorSpectrogramPlot();

    virtual void prepareForScan(const MotorScan s);
    MotorScan::MotorAxis leftAxis() const { return d_leftAxis; }
    MotorScan::MotorAxis bottomAxis() const { return d_bottomAxis; }

public slots:
    virtual void updateData(QVector<double> data, int cols);
    virtual void updatePoint(int row, int col, double val);
    virtual void setAxis(QwtPlot::Axis plotAxis, MotorScan::MotorAxis motorAxis);

protected:
    void filterData();
    void recalculateZRange();

    QwtMatrixRasterData *p_spectrogramData;
    QwtPlotSpectrogram *p_spectrogram;
    double d_max, d_min;
    bool d_firstPoint;
    MotorScan::MotorAxis d_leftAxis, d_bottomAxis;
    QMap<MotorScan::MotorAxis,QwtInterval> d_intervalList;


    // QwtPlot interface
public slots:
    virtual void replot();
};

#endif // MOTORSPECTROGRAMPLOT_H
