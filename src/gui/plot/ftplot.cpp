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
#include <data/experiment/overlaybase.h>

FtPlot::FtPlot(const QString &id, QWidget *parent) :
    ZoomPanPlot(BC::Key::ftPlot+id,parent), d_number(0), d_id(id)
{

    setPlotAxisTitle(QwtPlot::xBottom,QString("Frequency (MHz)"));
    setPlotAxisTitle(QwtPlot::yLeft,QString("FT "+id));

    // Disable QwtPlot's automatic memory management
    setAutoDelete(false);

    //build and configure curve object
    p_curve = CurveFactory::createStandardCurve<BlackchirpFTCurve>(BC::Key::ftCurve+id);
    attachCurve(p_curve.get());



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
    
    // Connect to curve metadata changes for overlay synchronization
    connect(this, &ZoomPanPlot::curveMetadataChanged, 
            this, &FtPlot::onCurveMetadataChanged);
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

    waitForFilterComplete();
    d_overlayCurves.clear();

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

void FtPlot::setMessageText(const QString &msg)
{
    auto text = p_messageLabel->text();
    text.setText(msg);
    p_messageLabel->setText(text);
}

void FtPlot::addOverlay(std::shared_ptr<OverlayBase> overlay)
{
    if (!overlay) {
        return;
    }

    // Check if this exact overlay object is already tracked. Identity
    // (not label) comparison matters here: a preview overlay can be
    // renamed after it was added to the plot (e.g. CatalogOverlayWidget
    // setting the label from the parsed molecule name), and the promotion
    // path in OverlayManagerWidget deliberately relies on this call
    // recognizing the just-promoted overlay as the same object already on
    // the plot rather than adding a second curve for it.
    for (const auto& pair : d_overlayCurves) {
        if (pair.first == overlay) {
            return;
        }
    }
    
    // Create overlay curve using CurveFactory
    QString curveKey = QString("overlay_%1").arg(overlay->getLabel());
    auto curve = CurveFactory::createOverlayCurve<BlackchirpPlotCurve>(curveKey, overlay);
    
    // Set curve data from overlay. The Y range is taken from the overlay
    // rather than measured off the points so that a region the overlay's
    // source suppresses does not stretch the axis.
    const auto [yMin,yMax] = overlay->displayYRange();
    curve->setCurveData(overlay->xyData(),yMin,yMax);
    curve->setTitle(overlay->getLabel());
    
    // Synchronize initial visibility with overlay enabled state
    curve->setCurveVisible(overlay->getEnabled());
    
    attachCurve(curve.get());

    // Store the overlay-curve pair
    d_overlayCurves.emplace_back(overlay, std::move(curve));
    
    replot();
}

void FtPlot::removeOverlay(std::shared_ptr<OverlayBase> overlay)
{
    if (!overlay) {
        return;
    }
    
    // Find and remove the overlay curve, matching by object identity (see
    // addOverlay() for why label matching is unsafe here).
    for (auto it = d_overlayCurves.begin(); it != d_overlayCurves.end(); ++it) {
        if (it->first == overlay) {
            // Drain any in-flight filter pass and unregister the curve while
            // it is still fully-typed. Relying on ~BlackchirpPlotCurveBase to
            // do this is too late: the derived destructor (and its _filter
            // override) is gone by the time the base destructor drains the
            // worker, so a concurrent pass would call a pure virtual.
            detachCurve(it->second.get());
            d_overlayCurves.erase(it);
            replot();
            return;
        }
    }
}

void FtPlot::updateOverlay(std::shared_ptr<OverlayBase> overlay)
{
    if (!overlay) {
        return;
    }
    
    // Find the existing overlay curve and update its data, matching by
    // object identity (see addOverlay() for why label matching is unsafe here).
    for (auto& pair : d_overlayCurves) {
        if (pair.first == overlay) {
            // Update curve data (applies scaling and offsets)
            const auto [yMin,yMax] = overlay->displayYRange();
            pair.second->setCurveData(overlay->xyData(),yMin,yMax);
            pair.second->setTitle(overlay->getLabel());
            
            // Update curve appearance from overlay metadata first
            pair.second->updateFromSettings();
            
            // Final visibility = overlay enabled state AND curve visibility from metadata
            bool curveVisibleFromMetadata = overlay->getCurveMetadata(BC::Key::bcCurveVisible).toBool();
            bool finalVisibility = overlay->getEnabled() && curveVisibleFromMetadata;
            pair.second->setCurveVisible(finalVisibility);
            
            replot();
            return;
        }
    }
}

bool FtPlot::hasOverlay(std::shared_ptr<OverlayBase> overlay) const
{
    if (!overlay) {
        return false;
    }
    
    // Check if overlay exists by object identity (see addOverlay() for why
    // label matching is unsafe here).
    for (const auto& pair : d_overlayCurves) {
        if (pair.first == overlay) {
            return true;
        }
    }
    return false;
}

void FtPlot::onCurveMetadataChanged(BlackchirpPlotCurveBase* curve)
{
    if (!curve) {
        return;
    }
    
    // Find if this curve belongs to an overlay
    for (auto& pair : d_overlayCurves) {
        if (pair.second.get() == curve) {
            // This curve belongs to an overlay - synchronize visibility
            bool curveVisible = curve->isVisible();
            bool overlayEnabled = pair.first->getEnabled();
            
            // Only update if there's a mismatch to avoid infinite loops
            if (curveVisible != overlayEnabled) {
                pair.first->setEnabled(curveVisible);
                // Emit signal to notify parent that overlay data changed
                emit overlayDataChanged(pair.first);
            }
            break;
        }
    }
}
