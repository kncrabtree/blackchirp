#ifndef MAINFTPLOT_H
#define MAINFTPLOT_H

#include "ftplot.h"
#include <QObject>

namespace BC::Key::FtMainPlot {
static const QString id{"Main"};
}

class MainFtPlot : public FtPlot
{
    Q_OBJECT
public:
    MainFtPlot(QWidget *parent = nullptr);
    
    virtual void prepareForExperiment(const Experiment &e);
    
public slots:
    void newPeakList(const QVector<QPointF> l);
    
private:
    BlackchirpPlotCurve *p_peakData;
    
};

#endif // MAINFTPLOT_H
