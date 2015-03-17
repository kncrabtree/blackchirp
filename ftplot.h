#ifndef FTPLOT_H
#define FTPLOT_H

#include "zoompanplot.h"
#include <QVector>
#include <QPointF>
#include <qwt6/qwt_plot_curve.h>
#include <qwt6/qwt_plot_zoomer.h>
#include <qwt6/qwt_plot_panner.h>
#include <qwt6/qwt_plot_picker.h>
#include <qwt6/qwt_plot_magnifier.h>
#include <qwt6/qwt_plot_grid.h>
#include <QTime>
#include "ftworker.h"
#include <QThread>

/*!
 * \brief Plot that displays the Fourier transform of the FID rolling average
 *
 * WARNING: DO NOT MESS WITH THIS CODE UNLESS YOU HAVE A GOOD UNDERSTANDING OF Qwt!!!
 * It is not for the faint of heart.
 *
 * This plot has several customizations on top of a standard QwtPlot.
 * The key improvement is data compression based on the number of horizontal pixels available; this greatly speeds up painting.
 * The FTPlot also has zooming, magnifying, and panning capabilities, but they are all customized to ensure the axes don't go beyond sensible limits.
 * Magnifying is done with the mouse wheel; hold Ctrl to magnify just the horizontal axis or Shift to magnify just the vertical axis.
 * Middle-click and drag to pan the plot.
 *
 */
class FtPlot : public ZoomPanPlot
{
    Q_OBJECT
public:

    /*!
     * \brief Initializes axes, etc. for the FT plot
     * \param parent Parent widget
     */
    explicit FtPlot(QWidget *parent = 0);
    ~FtPlot();

signals:
    void fidDone(const QVector<QPointF> fid);

public slots:
    void newFid(const Fid f);
    void ftDone(const QVector<QPointF> ft, double max);

    /*!
     * \brief Compresses data based on number of pixels wide the plot currently is
     *
     * This calculates the number of data points that will occupy a single pixel, based on the x-axis scale and width of plot.
     * It then loops over the most recent FFT data in chunks, finding the min and max y value in each.
     * In the end, each pixel has two data points: the min and max values of the data contained therein.
     * With 500000 points and 1000 pixels, or example, this compresses the data by 99.75% (2000/500000).
     */
    void filterData();

    void buildContextMenu(QPoint p);

    void ftColorSlot();
    void gridColorSlot();

    void ftStartChanged(double s);
    void ftEndChanged(double e);

    void updatePlot();



private:

    /*!
     * \brief The object representing the curve on the plot
     */
    QwtPlotCurve *p_curveData;
    QPair<double,double> d_autoScaleXRange, d_autoScaleYRange;

    QwtPlotGrid *p_plotGrid;

    QColor getColor(QColor startingColor);

    QThread *p_ftThread;
    FtWorker *p_ftw;
    Fid d_currentFid;
    QVector<QPointF> d_currentFt;

    bool d_processing;
    bool d_replotWhenDone;


};

#endif // FTPLOT_H
