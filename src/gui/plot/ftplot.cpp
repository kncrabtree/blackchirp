#include <gui/plot/ftplot.h>
#include <gui/plot/curvefactory.h>

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

    // Disable QwtPlot's automatic memory management
    setAutoDelete(false);

    //build and configure curve object
    p_curve = CurveFactory::createStandardCurve<BlackchirpFTCurve>(BC::Key::ftCurve+id);
    p_curve->attach(this);



    QPalette p;
    QColor bg( p.window().color() );
    bg.setAlpha( 232 );

    p_shotsLabel = std::make_unique<QwtPlotTextLabel>();
    QwtText text(d_shotsText.arg(0));
    text.setColor(p.text().color());
    text.setBackgroundBrush( QBrush( bg ) );
    text.setRenderFlags(Qt::AlignRight|Qt::AlignTop);
    p_shotsLabel->setText(text);
    p_shotsLabel->setZ(200.);
    p_shotsLabel->attach(this);

    p_messageLabel = std::make_unique<QwtPlotTextLabel>();
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
    // All items are managed by unique_ptr and will be automatically cleaned up
}

void FtPlot::prepareForExperiment(const Experiment &e)
{
    d_number = e.d_number;

    d_currentFt = Ft();
    p_curve->setCurrentFt(Ft());
    setNumShots(0);
    setMessageText("");

    p_curve->setVisible(e.ftmwEnabled());

    if(d_number>0)
        p_curve->setTitle(BC::Key::ftCurve+d_id+QString::number(d_number));
    else
        p_curve->setTitle(BC::Key::ftCurve+d_id);

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
