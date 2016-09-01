#ifndef MOTORSPECTROGRAMPLOT_H
#define MOTORSPECTROGRAMPLOT_H

#include "zoompanplot.h"

#include <qwt6/qwt_matrix_raster_data.h>
#include <qwt6/qwt_plot_spectrogram.h>

#include "motorscan.h"

class MotorSpectrogramPlot : public ZoomPanPlot
{
    Q_OBJECT
public:
    MotorSpectrogramPlot(QWidget *parent = nullptr);
    virtual ~MotorSpectrogramPlot();

    void setLabelText(QwtPlot::Axis axis, QString text);
    virtual void prepareForScan(const MotorScan s);
    virtual void buildContextMenu(QMouseEvent *me);
    MotorScan::MotorDataAxis leftAxis() const { return d_leftAxis; }
    MotorScan::MotorDataAxis bottomAxis() const { return d_bottomAxis; }

public slots:
    virtual void updateData(QVector<double> data, int cols);
    virtual void updatePoint(int row, int col, double val);
    virtual void setAxis(QwtPlot::Axis plotAxis, MotorScan::MotorDataAxis motorAxis);

protected:
    void filterData();
    void recalculateZRange();

    QwtMatrixRasterData *p_spectrogramData;
    QwtPlotSpectrogram *p_spectrogram;
    double d_max, d_min;
    bool d_firstPoint;
    MotorScan::MotorDataAxis d_leftAxis, d_bottomAxis;
    QMap<MotorScan::MotorDataAxis,QwtInterval> d_intervalList;


    // QwtPlot interface
public slots:
    virtual void replot();
};

#endif // MOTORSPECTROGRAMPLOT_H
