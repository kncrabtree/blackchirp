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

/// \brief Widget that edits the visual appearance of a single plot curve.
///
/// \sa CurveAppearance, BlackchirpPlotCurveBase, OverlayBase,
///     CurveAppearancePresetManager
class CurveAppearanceWidget : public QWidget
{
    Q_OBJECT

public:
    /// \brief Bundles all visual properties carried by a plot curve.
    ///
    /// \sa CurveAppearanceWidget::getCurrentAppearance,
    ///     CurveAppearanceWidget::setCurrentAppearance
    struct CurveAppearance {
        QColor color;                       ///< Pen color.
        QwtPlotCurve::CurveStyle curveStyle; ///< Rendering style (Lines, Sticks, Steps, Dots, NoCurve).
        double lineThickness;               ///< Pen width in pixels.
        Qt::PenStyle lineStyle;             ///< Line dash pattern (Qt::SolidLine, Qt::DashLine, etc.).
        QwtSymbol::Style markerStyle;       ///< Symbol drawn at each data point; QwtSymbol::NoSymbol suppresses markers.
        int markerSize;                     ///< Symbol size in pixels.
        bool visible;                       ///< Whether the curve is drawn on the plot.
        bool autoscale;                     ///< Whether the curve is included in autoscale range computation.
        QwtAxisId yAxis;                    ///< Y axis assignment (left or right).
    };

    /// \brief Constructs the widget with all appearance controls.
    /// \param parent Parent widget.
    explicit CurveAppearanceWidget(QWidget *parent = nullptr);

    /// \brief Destructor.
    ~CurveAppearanceWidget();

    /// \brief Populates all controls from the current settings of \a curve.
    /// \param curve Curve to read appearance from.
    void initializeFromCurve(BlackchirpPlotCurveBase *curve);

    /// \brief Writes the current control state to \a curve.
    /// \param curve Curve to update.
    void applyToCurve(BlackchirpPlotCurveBase *curve);

    /// \brief Populates all controls from the metadata stored in \a overlay.
    /// \param overlay Overlay whose metadata contains the serialized appearance.
    void initializeFromOverlay(std::shared_ptr<OverlayBase> overlay);

    /// \brief Writes the current control state into \a overlay's metadata.
    /// \param overlay Overlay to update.
    void applyToOverlay(std::shared_ptr<OverlayBase> overlay);

    /// \brief Returns the appearance struct reflecting the current control state.
    /// \return Current CurveAppearance.
    CurveAppearance getCurrentAppearance() const;

    /// \brief Sets all controls to match \a appearance.
    /// \param appearance Appearance values to apply to the controls.
    void setCurrentAppearance(const CurveAppearance &appearance);

    /// \brief Enables or disables the color-picker button.
    /// \param enabled Pass \c false to hide the color control (e.g., when color
    ///        is managed externally).
    void setColorButtonEnabled(bool enabled);

    /// \brief Enables or disables the Y-axis selector.
    /// \param enabled Pass \c false when the plot has only one Y axis.
    void setYAxisControlEnabled(bool enabled);

    /// \brief Refreshes the color swatch to show \a color without emitting signals.
    /// \param color New color to display.
    void updateColorDisplay(const QColor &color);

    /// \brief Attaches a preset manager so the preset controls become active.
    ///
    /// The widget does not take ownership of \a manager.
    /// \param manager Application-wide preset manager; pass \c nullptr to
    ///        disable preset controls.
    void setPresetManager(CurveAppearancePresetManager *manager);

    /// \brief Applies the named preset to all controls and emits
    ///        curveAppearanceChanged().
    /// \param presetName Name of the preset to apply.
    void applyPreset(const QString &presetName);

    /// \brief Saves the current control state as a preset named \a presetName.
    /// \param presetName Name under which to store the preset.
    void saveCurrentAsPreset(const QString &presetName);

    /// \brief Deletes the preset named \a presetName from the manager.
    /// \param presetName Name of the preset to remove.
    void deletePreset(const QString &presetName);

    /// \brief Repopulates the preset combo box from the attached manager.
    void refreshPresetList();

signals:
    /// \brief Emitted whenever any appearance control changes.
    /// \param appearance Updated appearance struct.
    void curveAppearanceChanged(const CurveAppearanceWidget::CurveAppearance &appearance);

    /// \brief Emitted when the user clicks the color button; callers may open
    ///        a color dialog and call updateColorDisplay() with the result.
    void colorChangeRequested();

    /// \brief Emitted when the user initiates a save; callers should prompt for
    ///        a name and call saveCurrentAsPreset().
    /// \param suggestedName Auto-generated name suggestion.
    void presetSaveRequested(const QString &suggestedName);

    /// \brief Emitted when the user initiates a delete; callers should confirm
    ///        before calling deletePreset().
    /// \param presetName Name of the preset the user wishes to remove.
    void presetDeleteRequested(const QString &presetName);

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

    CurveAppearance d_currentAppearance;
    bool d_blockSignals;
    CurveAppearancePresetManager *p_presetManager;
};

Q_DECLARE_METATYPE(CurveAppearanceWidget::CurveAppearance)

#endif // CURVEAPPEARANCEWIDGET_H
