#ifndef FTPLOT_H
#define FTPLOT_H

#include "zoompanplot.h"

#include <QVector>
#include <QPointF>

#include "experiment.h"

class QwtPlotCurve;
class QwtPlotGrid;

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

    void prepareForExperiment(const Experiment e);

signals:
    void pzfChanged(int);
    void unitsChanged(double);
    void winfChanged(BlackChirp::FtWindowFunction);

public slots:
    void newFt(const QVector<QPointF> ft, double max);
    void newFtDiff(const QVector<QPointF> ft, double min, double max);
    void filterData();
    void buildContextMenu(QMouseEvent *me);

    void changeFtColor(QColor c);
    void changeGridColor(QColor c);    
    void exportXY();
    void configureUnits(BlackChirp::FtPlotUnits u);
    void setWinf(BlackChirp::FtWindowFunction wf);
    void newPeakList(const QList<QPointF> l);


private:
    /*!
     * \brief The object representing the curve on the plot
     */
    QwtPlotCurve *p_curveData;
    QwtPlotCurve *p_peakData;
    QwtPlotGrid *p_plotGrid;
    QVector<QPointF> d_currentFt;
    int d_number;
    int d_pzf;
    BlackChirp::FtPlotUnits d_currentUnits;
    BlackChirp::FtWindowFunction d_currentWinf;

    QColor getColor(QColor startingColor);

};

#endif // FTPLOT_H
