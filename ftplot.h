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

    void buildContextMenu(QMouseEvent *me);

    void changeFtColor(QColor c);
    void changeGridColor(QColor c);

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
