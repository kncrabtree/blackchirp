#ifndef FTPLOT_H
#define FTPLOT_H

#include <gui/plot/zoompanplot.h>
#include <qwt6/qwt_plot_textlabel.h>

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

class BlackchirpFTCurve;

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

    void prepareForExperiment(const Experiment &e);
    Ft currentFt() const;

public slots:
    void newFt(const Ft ft);

    void configureUnits(FtWorker::FtUnits u);
    void newPeakList(const QVector<QPointF> l);
    void setNumShots(quint64 shots);


private:
    /*!
     * \brief The object representing the curve on the plot
     */
    BlackchirpPlotCurve *p_peakData;
    BlackchirpFTCurve *p_curve;
    QwtPlotTextLabel *p_label;

    Ft d_currentFt;
    int d_number;
    QString d_id;

    const QString d_shotsText{"Shots: %1"};

};

#endif // FTPLOT_H
