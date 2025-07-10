#ifndef CURVEAPPEARANCEWIDGET_H
#define CURVEAPPEARANCEWIDGET_H

#include <QWidget>
#include <QFormLayout>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QCheckBox>
#include <QPushButton>
#include <QLabel>

#include <qwt6/qwt_plot_curve.h>
#include <qwt6/qwt_symbol.h>
#include <qwt6/qwt_plot.h>

// Forward declaration
class BlackchirpPlotCurveBase;

class CurveAppearanceWidget : public QWidget
{
    Q_OBJECT
    
public:
    explicit CurveAppearanceWidget(QWidget *parent = nullptr);
    ~CurveAppearanceWidget();
    
    // Initialize the widget with curve data
    void initializeFromCurve(BlackchirpPlotCurveBase *curve);
    
    // Apply current settings to a curve
    void applyToCurve(BlackchirpPlotCurveBase *curve);
    
    // Get current appearance settings as a structure
    struct CurveAppearance {
        QColor color;
        QwtPlotCurve::CurveStyle curveStyle;
        double lineThickness;
        Qt::PenStyle lineStyle;
        QwtSymbol::Style markerStyle;
        int markerSize;
        bool visible;
        bool autoscale;
        QwtAxisId yAxis; // Using new QwtAxisId (QwtAxis::Position enum) - rest of codebase still uses QwtPlot::Axis
    };
    
    CurveAppearance getCurrentAppearance() const;
    void setCurrentAppearance(const CurveAppearance &appearance);
    
    // Enable/disable specific controls (useful for different contexts)
    void setColorButtonEnabled(bool enabled);
    void setYAxisControlEnabled(bool enabled);
    
    // Update color display when changed externally
    void updateColorDisplay(const QColor &color);
    
signals:
    void curveAppearanceChanged(const CurveAppearanceWidget::CurveAppearance &appearance);
    void colorChangeRequested(); // For opening color dialog externally if needed
    
private slots:
    void onColorButtonClicked();
    void onCurveStyleChanged(int index);
    void onLineThicknessChanged(double value);
    void onLineStyleChanged(int index);
    void onMarkerStyleChanged(int index);
    void onMarkerSizeChanged(int value);
    void onVisibilityChanged(bool visible);
    void onAutoscaleChanged(bool enabled);
    void onYAxisChanged(int index);
    
private:
    void setupUI();
    void setupConnections();
    void emitAppearanceChanged();
    
    // UI components
    QFormLayout *p_formLayout;
    QPushButton *p_colorButton;
    QComboBox *p_curveStyleBox;
    QDoubleSpinBox *p_thicknessBox;
    QComboBox *p_lineStyleBox;
    QComboBox *p_markerBox;
    QSpinBox *p_markerSizeBox;
    QCheckBox *p_visibleBox;
    QCheckBox *p_autoscaleBox;
    QComboBox *p_yAxisBox;
    
    // Current state
    CurveAppearance d_currentAppearance;
    bool d_blockSignals;
};

Q_DECLARE_METATYPE(CurveAppearanceWidget::CurveAppearance)

#endif // CURVEAPPEARANCEWIDGET_H