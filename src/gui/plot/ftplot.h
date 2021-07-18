#ifndef FTPLOT_H
#define FTPLOT_H

#include <gui/plot/zoompanplot.h>

#include <QVector>
#include <QPointF>

#include <data/experiment/experiment.h>
#include <data/analysis/ft.h>
#include <data/analysis/ftworker.h>

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

public slots:
    void newFt(const Ft ft);

    void configureUnits(FtWorker::FtUnits u);
    void newPeakList(const QList<QPointF> l);


private:
    /*!
     * \brief The object representing the curve on the plot
     */
    BlackchirpPlotCurve *p_curve, *p_peakData;

    Ft d_currentFt;
    int d_number;
    QString d_id;

};

#endif // FTPLOT_H
