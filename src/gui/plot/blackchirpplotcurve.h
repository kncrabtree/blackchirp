#ifndef BLACKCHIRPPLOTCURVE_H
#define BLACKCHIRPPLOTCURVE_H

#include <qwt6/qwt_plot.h>
#include <qwt6/qwt_plot_curve.h>
#include <qwt6/qwt_symbol.h>
#include <qwt6/qwt_text.h>
#include <qwt6/qwt_scale_map.h>

class QMutex;

Q_DECLARE_METATYPE(QwtSymbol::Style)
Q_DECLARE_METATYPE(QwtPlot::Axis)

#include <data/storage/settingsstorage.h>

namespace BC::Key {
static const QString bcCurve{"Curve"};
static const QString bcCurveColor{"color"};
static const QString bcCurveStyle{"style"};
static const QString bcCurveThickness{"thickness"};
static const QString bcCurveMarker{"marker"};
static const QString bcCurveMarkerSize{"markerSize"};
static const QString bcCurveAxisX{"xAxis"};
static const QString bcCurveAxisY{"yAxis"};
static const QString bcCurveVisible{"visible"};
static const QString bcCurvePlotIndex{"plotIndex"};
}

class BlackchirpPlotCurveBase : public QwtPlotCurve, public SettingsStorage
{
public:
    BlackchirpPlotCurveBase(const QString key, const QString title=QString(""),
                        Qt::PenStyle defaultLineStyle = Qt::SolidLine,
                        QwtSymbol::Style defaultMarker = QwtSymbol::NoSymbol);
    virtual ~BlackchirpPlotCurveBase();

    void setColor(const QColor c);
    void setLineThickness(double t);
    void setLineStyle(Qt::PenStyle s);
    void setMarkerStyle(QwtSymbol::Style s);
    void setMarkerSize(int s);
    QString name() const { return title().text(); }
    QString key() const { return d_key; }

    virtual QVector<QPointF> curveData() const =0;

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

    int plotIndex() const { return get(BC::Key::bcCurvePlotIndex,-1); }

    void updateFromSettings();
    void filter(int w, const QwtScaleMap map);

private:
    const QString d_key;
    QMutex *p_samplesMutex;


    void configurePen();
    void configureSymbol();
    void setSamples(const QVector<QPointF> d);

protected:
    virtual QVector<QPointF> _filter(int w, const QwtScaleMap map) =0;

    template<typename It>
    void increment(It i) { ++i; }

public:
    virtual QRectF boundingRect() const override =0;

    // QwtPlotItem interface
public:
    void draw(QPainter *painter, const QwtScaleMap &xMap, const QwtScaleMap &yMap, const QRectF &canvasRect) const override;
};

class BlackchirpPlotCurve : public BlackchirpPlotCurveBase
{
public:
    BlackchirpPlotCurve(const QString key, const QString title=QString(""),
                        Qt::PenStyle defaultLineStyle = Qt::SolidLine,
                        QwtSymbol::Style defaultMarker = QwtSymbol::NoSymbol);
    ~BlackchirpPlotCurve();

    void setCurveData(const QVector<QPointF> d);
    void setCurveData(const QVector<QPointF> d, double min, double max);
    void appendPoint(const QPointF p);

private:
    void calcBoundingRectHeight();

    QRectF d_boundingRect;
    QVector<QPointF> d_curveData;
    QMutex *p_dataMutex;

    // QwtPlotItem interface
public:
    QRectF boundingRect() const override;

    // BlackchirpPlotCurveBase interface
public:
    QVector<QPointF> curveData() const override;

protected:
    QVector<QPointF> _filter(int w, const QwtScaleMap map) override;
};

class BCEvenSpacedCurveBase : public BlackchirpPlotCurveBase
{
public:
    BCEvenSpacedCurveBase(const QString key, const QString title=QString(""),
                          Qt::PenStyle defaultLineStyle = Qt::SolidLine,
                          QwtSymbol::Style defaultMarker = QwtSymbol::NoSymbol);
    virtual ~BCEvenSpacedCurveBase();

    double xVal(int i) const;
    int indexBefore(double xVal) const;

protected:
    virtual double xFirst() const =0;
    virtual double spacing() const =0;
    virtual int numPoints() const =0;
    virtual QVector<double> yData() =0;
    QVector<QPointF> _filter(int w, const QwtScaleMap map) override final;
};

#include <data/analysis/ft.h>

class BlackchirpFTCurve : public BCEvenSpacedCurveBase
{
public:
    BlackchirpFTCurve(const QString key, const QString title=QString(""),
                      Qt::PenStyle defaultLineStyle = Qt::SolidLine,
                      QwtSymbol::Style defaultMarker = QwtSymbol::NoSymbol);
    ~BlackchirpFTCurve();

    void setCurrentFt(const Ft f);

private:
    QMutex *p_mutex;
    Ft d_currentFt;

    // QwtPlotItem interface
public:
    QRectF boundingRect() const override;

    // BlackchirpPlotCurveBase interface
public:
    QVector<QPointF> curveData() const override;

    // BCEvenSpacedCurveBase interface
private:
    double xFirst() const override;
    double spacing() const override;
    int numPoints() const override;
    QVector<double> yData() override;
};

class BlackchirpFIDCurve : public BCEvenSpacedCurveBase
{
public:
    BlackchirpFIDCurve(const QString key, const QString title=QString(""),
                       Qt::PenStyle defaultLineStyle = Qt::SolidLine,
                       QwtSymbol::Style defaultMarker = QwtSymbol::NoSymbol);
    ~BlackchirpFIDCurve();

    void setCurrentFid(const QVector<double> d, double spacing=1.0, double min=0.0, double max=0.0);

private:
    QMutex *p_mutex;
    QVector<double> d_fidData;
    double d_spacing{1.0}, d_min{0.0}, d_max{1.0};

    // QwtPlotItem interface
public:
    QRectF boundingRect() const override;

    // BlackchirpPlotCurveBase interface
public:
    QVector<QPointF> curveData() const override;

    // BCEvenSpacedCurveBase interface
protected:
    double xFirst() const override;
    double spacing() const override;
    int numPoints() const override;
    QVector<double> yData() override;
};

#endif // BLACKCHIRPPLOTCURVE_H
