#include "lifsliceplot.h"

#include <QSettings>
#include <QPalette>
#include <QFileDialog>
#include <QMessageBox>
#include <QMenu>

#include <qwt6/qwt_plot_curve.h>
#include <qwt6/qwt_plot_textlabel.h>
#include <qwt6/qwt_symbol.h>
#include <qwt6/qwt_scale_widget.h>

#include <gui/plot/blackchirpplotcurve.h>

LifSlicePlot::LifSlicePlot(const QString name, QWidget *parent) :
    ZoomPanPlot(name,parent)
{
    this->setPlotAxisTitle(QwtPlot::yLeft,QString("LIF (AU)"));

    p_curve = new BlackchirpPlotCurve(BC::Key::lifSliceCurve,"",Qt::SolidLine,QwtSymbol::Ellipse);
    p_curve->setZ(1.0);
    p_curve->attach(this);

}

LifSlicePlot::~LifSlicePlot()
{

}

void LifSlicePlot::prepareForExperiment()
{
    p_curve->setCurveData(QVector<QPointF>());
    autoScale();
}

void LifSlicePlot::setData(const QVector<QPointF> d)
{
    p_curve->setCurveData(d);
    replot();
}
