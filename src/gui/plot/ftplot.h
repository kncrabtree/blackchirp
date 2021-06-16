#ifndef FTPLOT_H
#define FTPLOT_H

#include <src/gui/plot/zoompanplot.h>

#include <QVector>
#include <QPointF>

#include <src/data/experiment/experiment.h>
#include <src/data/analysis/ft.h>

class QwtPlotCurve;
class QwtPlotGrid;

namespace BC::Key {
static const QString ftPlot("FtPlot");
static const QString ftCurve("FT");
static const QString peakCurve("FTPeaks");
}

class FtPlot : public ZoomPanPlot
{
    Q_OBJECT
public:

    /*!
     * \brief Initializes axes, etc. for the FT plot
     * \param parent Parent widget
     */
    explicit FtPlot(const QString id, QWidget *parent = 0);
    ~FtPlot();

    void prepareForExperiment(const Experiment e);
    Ft currentFt() const;

signals:
    void pzfChanged(int);
    void unitsChanged(double newScf);
    void scalingChange(double scfRatio);
    void winfChanged(BlackChirp::FtWindowFunction);

public slots:
    void newFt(const Ft ft);
    void filterData();
    void buildContextMenu(QMouseEvent *me);

    void changeGridColor(QColor c);
    void exportXY();
    void configureUnits(BlackChirp::FtPlotUnits u);
    void newPeakList(const QList<QPointF> l);


private:
    /*!
     * \brief The object representing the curve on the plot
     */
    BlackchirpPlotCurve *p_curve, *p_peakData;
    QwtPlotGrid *p_plotGrid;
    Ft d_currentFt;
    int d_number;
    QString d_id;
    BlackChirp::FtPlotUnits d_currentUnits;

};

#endif // FTPLOT_H
