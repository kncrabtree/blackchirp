#ifndef FTPLOT_H
#define FTPLOT_H

#include <gui/plot/zoompanplot.h>
#include <qwt6/qwt_plot_textlabel.h>
#include <memory>

#include <QVector>
#include <QPointF>

#include <data/experiment/experiment.h>
#include <data/analysis/ft.h>
#include <data/analysis/ftworker.h>

class OverlayBase;

namespace BC::Key {
static const QString ftPlot{"FtPlot"};
static const QString ftCurve{"FT"};
static const QString peakCurve{"FTPeaks"};
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

    virtual void prepareForExperiment(const Experiment &e);
    Ft currentFt() const;

public slots:
    void newFt(const Ft ft);

    void configureUnits(FtWorker::FtUnits u);
    
    void setNumShots(quint64 shots);
    void setMessageText(QString msg);
    
    void addOverlay(std::shared_ptr<OverlayBase> overlay);
    void removeOverlay(std::shared_ptr<OverlayBase> overlay);
    void updateOverlay(std::shared_ptr<OverlayBase> overlay);
    
protected:
    QString id() const { return d_id; }
    int number() const { return d_number; }


private:
    /*!
     * \brief The object representing the curve on the plot
     */
    std::unique_ptr<BlackchirpFTCurve> p_curve;
    std::unique_ptr<QwtPlotTextLabel> p_shotsLabel;
    std::unique_ptr<QwtPlotTextLabel> p_messageLabel;
    
    // Overlay curves
    std::vector<std::pair<std::shared_ptr<OverlayBase>, std::unique_ptr<BlackchirpPlotCurve>>> d_overlayCurves;

    Ft d_currentFt;
    int d_number;
    QString d_id;

    const QString d_shotsText{"Shots: %1"};

};

#endif // FTPLOT_H
