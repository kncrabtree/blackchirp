#ifndef FIDPLOT_H
#define FIDPLOT_H

#include <gui/plot/zoompanplot.h>
#include <qwt6/qwt_plot_textlabel.h>
#include <memory>

#include <QPointF>
#include <QPair>

#include <data/experiment/experiment.h>

class QwtPlotMarker;

namespace BC::Key {
inline constexpr QLatin1StringView fidPlot{"FidPlot"};
inline constexpr QLatin1StringView fidCurve{"FID"};
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
    ~FidPlot();

signals:
    void ftStartChanged(double);
    void ftEndChanged(double);


public slots:
    void receiveProcessedFid(const QVector<double> d, double spacing, double min, double max, quint64 shots);
    void prepareForExperiment(const Experiment &e);
    void setFtStart(double start);
    void setFtEnd(double end);
    void setNumShots(quint64 shots);

private:
    std::unique_ptr<BlackchirpFIDCurve> p_curve;
    std::unique_ptr<QwtPlotTextLabel> p_label;

    QPair<std::unique_ptr<QwtPlotMarker>,std::unique_ptr<QwtPlotMarker>> d_chirpMarkers;
    QPair<std::unique_ptr<QwtPlotMarker>,std::unique_ptr<QwtPlotMarker>> d_ftMarkers;

    const QString d_shotsText{"Shots: %1"};

};


#endif // FIDPLOT_H
