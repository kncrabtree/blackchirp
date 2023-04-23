#ifndef LIFSPECTROGRAMPLOT_H
#define LIFSPECTROGRAMPLOT_H

#include <gui/plot/zoompanplot.h>

#include <modules/lif/data/lifconfig.h>

class QwtPlotSpectrogram;
class QwtMatrixRasterData;
class QwtPlotMarker;
class QMouseEvent;

namespace BC::Key {
static const QString lifSpectrogram{"lifSpectrogram"};
}

class LifSpectrogramPlot : public ZoomPanPlot
{
    Q_OBJECT
public:
    LifSpectrogramPlot(QWidget *parent = nullptr);
    ~LifSpectrogramPlot();

    void clear();
    void prepareForExperiment(const LifConfig &c);
    void updateData(const QVector<double> d, int numCols);
    void setLiveIndices(int di, int li);

    void setZMax(double d);
    void replot();

    double delayVal(int index) const;
    double laserVal(int index) const;

    int currentDelayIndex() const;
    int currentLaserIndex() const;

    double getdMin() const;
    double getddx() const;
    double getlMin() const;
    double getldx() const;

public slots:
    void moveLaserCursor(QPoint pos);
    void moveLaserCursor(int index);
    void moveDelayCursor(QPoint pos);
    void moveDelayCursor(int index);
    void buildContextMenu(QMouseEvent *me);

signals:
    void laserSlice(int delayIndex);
    void delaySlice(int freqIndex);

private:
    QwtMatrixRasterData *p_spectrogramData;
    QwtPlotSpectrogram *p_spectrogram;
    QwtPlotMarker *p_delayMarker, *p_laserMarker;
    bool d_enabled, d_live{true};
    bool d_delayDragging, d_freqDragging, d_grabDelay, d_grabFreq;
    int d_liveDelayIndex{0}, d_liveLaserIndex{0};

    double d_dMin, d_ddx, d_lMin, d_ldx;


    // ZoomPanPlot interface
protected:
    bool eventFilter(QObject *obj, QEvent *ev);
};

#endif // LIFSPECTROGRAMPLOT_H
