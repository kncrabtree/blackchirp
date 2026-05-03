#include "mainftplot.h"
#include <gui/plot/curvefactory.h>

#include <gui/plot/blackchirpplotcurve.h>

MainFtPlot::MainFtPlot(QWidget *parent) :
    FtPlot(BC::Key::FtMainPlot::id,parent)
{
    p_peakData = CurveFactory::createStandardCurve<BlackchirpPlotCurve>(BC::Key::peakCurve+id(),SettingsStorage::General,"",Qt::NoPen,QwtSymbol::Ellipse);
    attachCurve(p_peakData.get());
}

MainFtPlot::~MainFtPlot()
{
    // Base class FtPlot::~FtPlot() will handle detachItems()
}

void MainFtPlot::prepareForExperiment(const Experiment &e)
{
    FtPlot::prepareForExperiment(e);
    
    p_peakData->setCurveData(QVector<QPointF>());
    
    if(number()>0)
        p_peakData->setTitle(BC::Key::peakCurve+id()+QString::number(number()));
    else
        p_peakData->setTitle(BC::Key::peakCurve+id());

}

void MainFtPlot::newPeakList(const QVector<QPointF> l)
{
    
    if(!l.isEmpty())
        p_peakData->setCurveData(l);

    p_peakData->setCurveVisible(!l.isEmpty());
    replot();
}
