#ifndef CURVEAPPEARANCEWIDGET_H
#define CURVEAPPEARANCEWIDGET_H

#include <QWidget>
#include <QVBoxLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QCheckBox>
#include <QPushButton>
#include <QLabel>
#include <memory>

#include <qwt6/qwt_plot_curve.h>
#include <qwt6/qwt_symbol.h>
#include <qwt6/qwt_plot.h>

// Forward declarations
class BlackchirpPlotCurveBase;
class OverlayBase;
class CurveAppearancePresetManager;

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
    
    // Initialize the widget from overlay metadata
    void initializeFromOverlay(std::shared_ptr<OverlayBase> overlay);
    
    // Apply current settings to overlay metadata
    void applyToOverlay(std::shared_ptr<OverlayBase> overlay);
    
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
    
    // Preset management
    void setPresetManager(CurveAppearancePresetManager *manager);
    void applyPreset(const QString &presetName);
    void saveCurrentAsPreset(const QString &presetName);
    void deletePreset(const QString &presetName);
    void refreshPresetList();
    
signals:
    void curveAppearanceChanged(const CurveAppearanceWidget::CurveAppearance &appearance);
    void colorChangeRequested(); // For opening color dialog externally if needed
    void presetSaveRequested(const QString &suggestedName); // Request user input for preset name
    void presetDeleteRequested(const QString &presetName); // Request confirmation for preset deletion
    
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
    
    // Preset-related slots
    void onPresetSelected(int index);
    void onSavePresetClicked();
    void onDeletePresetClicked();
    
private:
    void setupUI();
    void setupConnections();
    void emitAppearanceChanged();
    void updateDeleteButtonState();
    QString generatePresetSuggestion() const;
    QString getMarkerShapeName(QwtSymbol::Style style) const;
    QString getLineStyleName(Qt::PenStyle style) const;
    QString getColorDescription(const QColor &color) const;
    
    // UI components - using VBoxLayout with GroupBoxes instead of FormLayout
    
    // Preset controls (at top)
    QComboBox *p_presetBox;
    QPushButton *p_savePresetButton;
    QPushButton *p_deletePresetButton;
    
    // Appearance controls
    QPushButton *p_colorButton;
    QComboBox *p_curveStyleBox;
    QDoubleSpinBox *p_thicknessBox;
    QComboBox *p_lineStyleBox;
    QComboBox *p_markerBox;
    QSpinBox *p_markerSizeBox;
    QCheckBox *p_visibleBox;
    QCheckBox *p_autoscaleBox;
    QComboBox *p_yAxisBox;
    
    // Current state and preset management
    CurveAppearance d_currentAppearance;
    bool d_blockSignals;
    CurveAppearancePresetManager *p_presetManager;
};

Q_DECLARE_METATYPE(CurveAppearanceWidget::CurveAppearance)

#endif // CURVEAPPEARANCEWIDGET_H