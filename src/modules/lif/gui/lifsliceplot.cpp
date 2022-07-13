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

    p_label = new QwtPlotTextLabel();
    p_label->setZ(10.0);
    p_label->setItemAttribute(QwtPlotItem::AutoScale,false);
    p_label->attach(this);

}

LifSlicePlot::~LifSlicePlot()
{

}

void LifSlicePlot::prepareForExperiment()
{
    p_curve->setCurveData(QVector<QPointF>());
    autoScale();
}

void LifSlicePlot::setData(const QVector<QPointF> d, QString txt)
{
    QwtText t;

    t.setRenderFlags(Qt::AlignRight | Qt::AlignTop);
    t.setBackgroundBrush(QBrush(QPalette().color(QPalette::Window)));
    QColor border = QPalette().color(QPalette::Text);
    border.setAlpha(0);
    t.setBorderPen(QPen(border));
    t.setColor(QPalette().color(QPalette::Text));

    t.setText(txt);
    p_label->setText(t);

    p_curve->setCurveData(d);
    replot();
}
