#include <gui/plot/fidplot.h>
#include <gui/plot/curvefactory.h>
#include <math.h>

#include <QPalette>
#include <QApplication>
#include <QMenu>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QWidgetAction>
#include <QFormLayout>
#include <QColorDialog>
#include <QMouseEvent>
#include <QFileDialog>
#include <QMessageBox>

#include <qwt6/qwt_plot_canvas.h>
#include <qwt6/qwt_plot_marker.h>
#include <gui/plot/blackchirpplotcurve.h>

FidPlot::FidPlot(const QString id, QWidget *parent) :
    ZoomPanPlot(BC::Key::fidPlot+id,parent)
{

    setPlotAxisTitle(QwtPlot::xBottom,QString::fromUtf16(u"Time (μs)"));
    setPlotAxisTitle(QwtPlot::yLeft,QString("FID ")+id);

    // Disable QwtPlot's automatic memory management
    setAutoDelete(false);

    p_curve = CurveFactory::createStandardCurve<BlackchirpFIDCurve>(BC::Key::fidCurve+id);
    p_curve->attach(this);

    d_chirpMarkers.first = std::make_unique<QwtPlotMarker>();
    d_chirpMarkers.first->setLineStyle(QwtPlotMarker::VLine);
    d_chirpMarkers.first->setLinePen(QPen(QPalette().color(QPalette::Text)));
    QwtText csLabel(QString("Chirp Start"));
    csLabel.setFont(QFont(QString("sans serif"),6));
    d_chirpMarkers.first->setLabel(csLabel);
    d_chirpMarkers.first->setLabelOrientation(Qt::Vertical);
    d_chirpMarkers.first->setLabelAlignment(Qt::AlignTop|Qt::AlignRight);
    d_chirpMarkers.first->attach(this);
    d_chirpMarkers.first->setVisible(false);

    d_chirpMarkers.second = std::make_unique<QwtPlotMarker>();
    d_chirpMarkers.second->setLineStyle(QwtPlotMarker::VLine);
    d_chirpMarkers.second->setLinePen(QPen(QPalette().color(QPalette::Text)));
    QwtText ceLabel(QString("Chirp End"));
    ceLabel.setFont(QFont(QString("sans serif"),6));
    d_chirpMarkers.second->setLabel(ceLabel);
    d_chirpMarkers.second->setLabelOrientation(Qt::Vertical);
    d_chirpMarkers.second->setLabelAlignment(Qt::AlignBottom|Qt::AlignLeft);
    d_chirpMarkers.second->attach(this);
    d_chirpMarkers.second->setVisible(false);


    d_ftMarkers.first = std::make_unique<QwtPlotMarker>();
    d_ftMarkers.first->setLineStyle(QwtPlotMarker::VLine);
    d_ftMarkers.first->setLinePen(QPen(QPalette().color(QPalette::Text)));
    QwtText ftsLabel(QString(" FT Start "));
    ftsLabel.setFont(QFont(QString("sans serif"),6));
    ftsLabel.setBackgroundBrush(QPalette().window());
    ftsLabel.setColor(QPalette().text().color());
    d_ftMarkers.first->setLabel(ftsLabel);
    d_ftMarkers.first->setLabelOrientation(Qt::Vertical);
    d_ftMarkers.first->setLabelAlignment(Qt::AlignTop|Qt::AlignRight);
    d_ftMarkers.first->setXValue(0.0);
    d_ftMarkers.first->attach(this);
    d_ftMarkers.first->setVisible(false);

    d_ftMarkers.second = std::make_unique<QwtPlotMarker>();
    d_ftMarkers.second->setLineStyle(QwtPlotMarker::VLine);
    d_ftMarkers.second->setLinePen(QPen(QPalette().color(QPalette::Text)));
    QwtText fteLabel(QString(" FT End "));
    fteLabel.setFont(QFont(QString("sans serif"),6));
    fteLabel.setBackgroundBrush(QPalette().window());
    fteLabel.setColor(QPalette().text().color());
    d_ftMarkers.second->setLabel(fteLabel);
    d_ftMarkers.second->setLabelOrientation(Qt::Vertical);
    d_ftMarkers.second->setLabelAlignment(Qt::AlignBottom|Qt::AlignLeft);
    d_ftMarkers.second->setXValue(0.0);
    d_ftMarkers.second->attach(this);
    d_ftMarkers.second->setVisible(false);

    QPalette p;
    QColor bg( p.window().color() );
    bg.setAlpha( 232 );

    p_label = std::make_unique<QwtPlotTextLabel>();
    QwtText text(d_shotsText.arg(0));
    text.setColor(p.text().color());
    text.setBackgroundBrush( QBrush( bg ) );
    text.setRenderFlags(Qt::AlignRight|Qt::AlignTop);
    p_label->setZ(200.);
    p_label->setText(text);
    p_label->attach(this);



}

FidPlot::~FidPlot()
{
    // All items are managed by unique_ptr and will be automatically cleaned up
}

void FidPlot::receiveProcessedFid(const QVector<double> d, double spacing, double min, double max, quint64 shots)
{
    p_curve->setCurrentFid(d,spacing,min,max);
    setNumShots(shots);

    replot();
}

void FidPlot::prepareForExperiment(const Experiment &e)
{     
    if(!e.ftmwEnabled())
    {
        p_curve->setVisible(false);

        d_chirpMarkers.first->setVisible(false);
        d_chirpMarkers.second->setVisible(false);
        d_ftMarkers.first->setVisible(false);
        d_ftMarkers.second->setVisible(false);

        autoScale();
        return;
    }

    auto c = e.ftmwConfig();
    p_curve->setCurrentFid({});

    p_curve->setVisible(true);

    d_ftMarkers.first->setVisible(true);
    d_ftMarkers.second->setVisible(true);

    double maxTime = (static_cast<double>(c->scopeConfig().d_recordLength)-1.0)/c->scopeConfig().d_sampleRate*1e6;
    double ftEnd = d_ftMarkers.second->xValue();
    if(ftEnd <= 0.0 || ftEnd <= d_ftMarkers.first->xValue() || ftEnd > maxTime)
        d_ftMarkers.second->setXValue(maxTime);

    emit ftStartChanged(d_ftMarkers.first->xValue());
    emit ftEndChanged(d_ftMarkers.second->xValue());

    bool displayMarkers = c->d_phaseCorrectionEnabled || c->d_chirpScoringEnabled;
    if(displayMarkers)
    {
        auto r = c->chirpRange();
        double chirpStart = (double)r.first*1e6/c->scopeConfig().d_sampleRate;
        double chirpEnd = chirpStart + (double)r.second*1e6/c->scopeConfig().d_sampleRate;

        d_chirpMarkers.first->setValue(chirpStart,0.0);
        d_chirpMarkers.second->setValue(chirpEnd,0.0);
    }

    d_chirpMarkers.first->setVisible(displayMarkers);
    d_chirpMarkers.second->setVisible(displayMarkers);

    autoScale();
}

void FidPlot::setFtStart(double start)
{
    double v = start;
    auto currentFid = p_curve->curveData();
    if(!currentFid.isEmpty())
        v = qBound(0.0,start,qMin(d_ftMarkers.second->value().x(),currentFid.constLast().x()*1e6));

    d_ftMarkers.first->setValue(v,0.0);

    emit ftStartChanged(v);

    replot();
}

void FidPlot::setFtEnd(double end)
{
    double v = end;
    auto currentFid = p_curve->curveData();
    if(!currentFid.isEmpty())
        v = qBound(d_ftMarkers.first->value().x(),end,currentFid.constLast().x()*1e6);

    d_ftMarkers.second->setValue(v,0.0);
    emit ftEndChanged(v);

    replot();
}

void FidPlot::setNumShots(quint64 shots)
{
    auto text = p_label->text();
    text.setText(d_shotsText.arg(shots));
    p_label->setText(text);
}
