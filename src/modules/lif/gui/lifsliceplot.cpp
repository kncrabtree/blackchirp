#include "lifsliceplot.h"
#include <gui/plot/curvefactory.h>

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

    // Disable QwtPlot's automatic memory management
    setAutoDelete(false);

    p_curve = CurveFactory::createStandardCurve<BlackchirpPlotCurve>(BC::Key::lifSliceCurve,"",Qt::SolidLine,QwtSymbol::Ellipse);
    p_curve->setZ(1.0);
    p_curve->attach(this);

    p_label = std::make_unique<QwtPlotTextLabel>();
    p_label->setZ(10.0);
    p_label->setItemAttribute(QwtPlotItem::AutoScale,false);
    p_label->attach(this);

}

LifSlicePlot::~LifSlicePlot()
{
    // All items are managed by unique_ptr and will be automatically cleaned up
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
