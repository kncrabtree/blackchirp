#include <gui/plot/ftplot.h>

#include <QFont>
#include <QMouseEvent>
#include <QEvent>
#include <QMenu>
#include <QAction>
#include <QActionGroup>
#include <QColorDialog>
#include <QApplication>
#include <QWidgetAction>
#include <QSpinBox>
#include <QFormLayout>
#include <QFileDialog>
#include <QMessageBox>
#include <QList>
#include <QPair>
#include <QDialogButtonBox>
#include <QPushButton>


#include <qwt6/qwt_scale_widget.h>
#include <qwt6/qwt_scale_div.h>
#include <qwt6/qwt_plot_grid.h>


#include <gui/plot/blackchirpplotcurve.h>

FtPlot::FtPlot(const QString id, QWidget *parent) :
    ZoomPanPlot(BC::Key::ftPlot+id,parent), d_number(0), d_id(id)
{

    setPlotAxisTitle(QwtPlot::xBottom,QString("Frequency (MHz)"));
    setPlotAxisTitle(QwtPlot::yLeft,QString("FT "+id));

    //build and configure curve object
    p_curve = new BlackchirpPlotCurve(BC::Key::ftCurve+id);
    p_curve->attach(this);

    p_peakData = new BlackchirpPlotCurve(BC::Key::peakCurve+id,"",Qt::NoPen,QwtSymbol::Ellipse);
    p_peakData->attach(this);
}

FtPlot::~FtPlot()
{
}

void FtPlot::prepareForExperiment(const Experiment e)
{
    d_number = e.d_number;

    d_currentFt = Ft();
    p_curve->setCurveData(QVector<QPointF>());
    p_peakData->setCurveData(QVector<QPointF>());

    p_curve->setVisible(e.ftmwEnabled());

    autoScale();
}

Ft FtPlot::currentFt() const
{
    return d_currentFt;
}

void FtPlot::newFt(const Ft ft)
{
    d_currentFt = ft;
    p_curve->setCurveData(ft.toVector(),ft.yMin(),ft.yMax());
    replot();
}


void FtPlot::configureUnits(FtWorker::FtUnits u)
{

    QwtText title = axisTitle(QwtPlot::yLeft);

    switch(u)
    {
    case FtWorker::FtV:
        title.setText(QString("FT "+d_id+" (V)"));
        break;
    case FtWorker::FtmV:
        title.setText(QString("FT "+d_id+" (mV)"));
        break;
    case FtWorker::FtuV:
        title.setText(QString("FT "+d_id+QString::fromUtf16(u" (µV)")));
        break;
    case FtWorker::FtnV:
        title.setText(QString("FT "+d_id+" (nV)"));
        break;
    default:
        break;
    }



    setAxisTitle(QwtPlot::yLeft,title);
//    emit unitsChanged(scf);
//    emit scalingChange(scf/oldScf);
}

void FtPlot::newPeakList(const QList<QPointF> l)
{

    if(!l.isEmpty())
        p_peakData->setCurveData(l.toVector());

    p_peakData->setCurveVisible(!l.isEmpty());
    replot();
}
