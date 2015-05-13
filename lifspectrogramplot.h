#ifndef LIFSPECTROGRAMPLOT_H
#define LIFSPECTROGRAMPLOT_H

#include "zoompanplot.h"

class QwtPlotSpectrogram;
class QwtMatrixRasterData;
class QwtLinearColorMap;

class LifSpectrogramPlot : public ZoomPanPlot
{
    Q_OBJECT
public:
    LifSpectrogramPlot(QWidget *parent = nullptr);
    ~LifSpectrogramPlot();

    void setRasterData(QwtMatrixRasterData *dat);
    void prepareForExperiment(double xMin, double xMax, double yMin, double yMax, bool enabled = true);

    void setZMax(double d);
    void replot();

private:
    QwtPlotSpectrogram *p_spectrogram;
    bool d_enabled;
    double d_zMax;


    // ZoomPanPlot interface
protected:
    void filterData();
};

#endif // LIFSPECTROGRAMPLOT_H
