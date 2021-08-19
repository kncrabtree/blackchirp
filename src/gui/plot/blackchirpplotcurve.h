#ifndef BLACKCHIRPPLOTCURVE_H
#define BLACKCHIRPPLOTCURVE_H

#include <qwt6/qwt_plot.h>
#include <qwt6/qwt_plot_curve.h>
#include <qwt6/qwt_symbol.h>

class QMutex;

Q_DECLARE_METATYPE(QwtSymbol::Style)
Q_DECLARE_METATYPE(QwtPlot::Axis)

#include <data/storage/settingsstorage.h>

namespace BC::Key {
static const QString bcCurve("Curve");
static const QString bcCurveColor("color");
static const QString bcCurveStyle("style");
static const QString bcCurveThickness("thickness");
static const QString bcCurveMarker("marker");
static const QString bcCurveMarkerSize("markerSize");
static const QString bcCurveAxisX("xAxis");
static const QString bcCurveAxisY("yAxis");
static const QString bcCurveVisible("visible");
static const QString bcCurvePlotIndex("plotIndex");
}

class BlackchirpPlotCurve : public QwtPlotCurve, public SettingsStorage
{
public:
    BlackchirpPlotCurve(const QString key, const QString title=QString(""),
                        Qt::PenStyle defaultLineStyle = Qt::SolidLine,
                        QwtSymbol::Style defaultMarker = QwtSymbol::NoSymbol);
    virtual ~BlackchirpPlotCurve();

    void setColor(const QColor c);
    void setLineThickness(double t);
    void setLineStyle(Qt::PenStyle s);
    void setMarkerStyle(QwtSymbol::Style s);
    void setMarkerSize(int s);
    void setCurveData(const QVector<QPointF> d);
    void setCurveData(const QVector<QPointF> d, double min, double max);
    void appendPoint(const QPointF p);
    QVector<QPointF> curveData() const { return d_curveData; }
    QString name() const { return title().text(); }
    QString key() const { return d_key; }

    /*!
     * \brief Sets curve visibility, and stores to settings
     *
     * Note: if you do not wish to save to settings, use setVisible directly.
     * This function is mainly intended for use with Tracking/Logging plots.
     *
     * \param v Curve visibility
     */
    void setCurveVisible(bool v);

    void setCurveAxisX(QwtPlot::Axis a);
    void setCurveAxisY(QwtPlot::Axis a);
    void setCurvePlotIndex(int i);

    int plotIndex() const { return get<int>(BC::Key::bcCurvePlotIndex,0); }

    void updateFromSettings();
    void filter(int w, const QwtScaleMap map);

private:
    const QString d_key;
    QVector<QPointF> d_curveData;
    QRectF d_boundingRect;
    QMutex *p_samplesMutex;
    QMutex *p_dataMutex;

    void configurePen();
    void configureSymbol();
    void setSamples(const QVector<QPointF> d);
    void calcBoundingRectHeight();



    // QwtPlotItem interface
public:
    QRectF boundingRect() const override;

    // QwtPlotItem interface
public:
    void draw(QPainter *painter, const QwtScaleMap &xMap, const QwtScaleMap &yMap, const QRectF &canvasRect) const override;
};

#endif // BLACKCHIRPPLOTCURVE_H
