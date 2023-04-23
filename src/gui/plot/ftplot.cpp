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
#include <QVector>
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
    p_curve = new BlackchirpFTCurve(BC::Key::ftCurve+id);
    p_curve->attach(this);

    p_peakData = new BlackchirpPlotCurve(BC::Key::peakCurve+id,"",Qt::NoPen,QwtSymbol::Ellipse);
    p_peakData->attach(this);

    QPalette p;
    QColor bg( p.window().color() );
    bg.setAlpha( 232 );

    p_shotsLabel = new QwtPlotTextLabel;
    QwtText text(d_shotsText.arg(0));
    text.setColor(p.text().color());
    text.setBackgroundBrush( QBrush( bg ) );
    text.setRenderFlags(Qt::AlignRight|Qt::AlignTop);
    p_shotsLabel->setText(text);
    p_shotsLabel->setZ(200.);
    p_shotsLabel->attach(this);

    p_messageLabel = new QwtPlotTextLabel;
    QwtText msg;
    msg.setColor(p.text().color());
    msg.setBackgroundBrush( QBrush( bg ) );
    msg.setRenderFlags(Qt::AlignLeft|Qt::AlignTop);
    p_messageLabel->setText(msg);
    p_messageLabel->setZ(200.);
    p_messageLabel->attach(this);
}

FtPlot::~FtPlot()
{
}

void FtPlot::prepareForExperiment(const Experiment &e)
{
    d_number = e.d_number;

    d_currentFt = Ft();
    p_curve->setCurrentFt(Ft());
    p_peakData->setCurveData(QVector<QPointF>());
    setNumShots(0);
    setMessageText("");

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
    p_curve->setCurrentFt(ft);
    setNumShots(ft.shots());
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
}

void FtPlot::newPeakList(const QVector<QPointF> l)
{

    if(!l.isEmpty())
        p_peakData->setCurveData(l);

    p_peakData->setCurveVisible(!l.isEmpty());
    replot();
}

void FtPlot::setNumShots(quint64 shots)
{
    auto text = p_shotsLabel->text();
    text.setText(d_shotsText.arg(shots));
    p_shotsLabel->setText(text);
}

void FtPlot::setMessageText(QString msg)
{
    auto text = p_messageLabel->text();
    text.setText(msg);
    p_messageLabel->setText(text);
}
