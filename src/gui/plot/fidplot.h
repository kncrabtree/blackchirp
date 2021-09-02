#ifndef FIDPLOT_H
#define FIDPLOT_H

#include <gui/plot/zoompanplot.h>
#include <qwt6/qwt_plot_textlabel.h>

#include <QPointF>
#include <QPair>

#include <data/experiment/experiment.h>

class QwtPlotMarker;

namespace BC::Key {
static const QString fidPlot{"FidPlot"};
static const QString fidCurve{"FID"};
}

class BlackchirpFIDCurve;

/*!
 * \brief The FID Plot
 */
class FidPlot : public ZoomPanPlot
{
    Q_OBJECT
public:

    /*!
     * \brief Constructor. Initializes axes and plot options
     * \param parent Parent widget
     */
    explicit FidPlot(const QString id, QWidget *parent = 0);

signals:
    void ftStartChanged(double);
    void ftEndChanged(double);


public slots:
    void receiveProcessedFid(const QVector<double> d, double spacing, double min, double max);
    void prepareForExperiment(const Experiment &e);
    void setFtStart(double start);
    void setFtEnd(double end);
    void setNumShots(quint64 shots);

private:
    BlackchirpFIDCurve *p_curve;
    QwtPlotTextLabel *p_label;

    QPair<QwtPlotMarker*,QwtPlotMarker*> d_chirpMarkers;
    QPair<QwtPlotMarker*,QwtPlotMarker*> d_ftMarkers;

    const QString d_shotsText{"Shots: %1"};

};


#endif // FIDPLOT_H
