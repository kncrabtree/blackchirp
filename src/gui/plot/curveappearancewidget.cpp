#include "curveappearancewidget.h"
#include "curveappearancepresetmanager.h"
#include "blackchirpplotcurve.h"
#include <data/experiment/overlaybase.h>

#include <QColorDialog>
#include <QLabel>
#include <QInputDialog>
#include <QMessageBox>
#include <QHBoxLayout>

CurveAppearanceWidget::CurveAppearanceWidget(QWidget *parent)
    : QWidget(parent), d_blockSignals(false), p_presetManager(nullptr)
{
    setupUI();
    setupConnections();
    
    // Connect to global preset manager
    setPresetManager(CurveAppearancePresetManager::instance());
    
    // Initialize with default appearance
    d_currentAppearance.color = palette().color(QPalette::Text);
    d_currentAppearance.curveStyle = QwtPlotCurve::Lines;
    d_currentAppearance.lineThickness = 1.0;
    d_currentAppearance.lineStyle = Qt::SolidLine;
    d_currentAppearance.markerStyle = QwtSymbol::NoSymbol;
    d_currentAppearance.markerSize = 7;
    d_currentAppearance.visible = true;
    d_currentAppearance.autoscale = true;
    d_currentAppearance.yAxis = QwtAxis::YLeft;
    
    // Set initial color button appearance
    QColor borderColor = palette().color(QPalette::Text);
    QColor textColor = (d_currentAppearance.color.lightness() > 128) ? Qt::black : Qt::white;
    QString colorStyle = QString("background-color: %1; border: 1px solid %2; color: %3;")
                        .arg(d_currentAppearance.color.name())
                        .arg(borderColor.name())
                        .arg(textColor.name());
    p_colorButton->setStyleSheet(colorStyle);
}

CurveAppearanceWidget::~CurveAppearanceWidget()
{
}

void CurveAppearanceWidget::setupUI()
{
    p_formLayout = new QFormLayout(this);
    
    // === PRESET CONTROLS (at top) ===
    // Preset selection combo box
    p_presetBox = new QComboBox(this);
    p_presetBox->setToolTip("Select a preset to apply or create a new preset");
    p_formLayout->addRow("Preset:", p_presetBox);
    
    // Preset action buttons in horizontal layout
    QWidget *presetButtonWidget = new QWidget(this);
    QHBoxLayout *presetButtonLayout = new QHBoxLayout(presetButtonWidget);
    presetButtonLayout->setContentsMargins(0, 0, 0, 0);
    
    p_savePresetButton = new QPushButton("Save...", this);
    p_savePresetButton->setToolTip("Save current appearance as a new preset");
    p_savePresetButton->setMaximumWidth(80);
    
    p_deletePresetButton = new QPushButton("Delete", this);
    p_deletePresetButton->setToolTip("Delete the selected preset");
    p_deletePresetButton->setMaximumWidth(80);
    p_deletePresetButton->setEnabled(false);
    
    presetButtonLayout->addWidget(p_savePresetButton);
    presetButtonLayout->addWidget(p_deletePresetButton);
    presetButtonLayout->addStretch();
    
    p_formLayout->addRow("", presetButtonWidget);
    
    // Add a separator line
    QLabel *separator = new QLabel(this);
    separator->setFrameStyle(QFrame::HLine | QFrame::Sunken);
    p_formLayout->addRow(separator);
    
    // === APPEARANCE CONTROLS ===
    // Color button
    p_colorButton = new QPushButton(this);
    p_colorButton->setText("Choose Color...");
    p_colorButton->setMinimumHeight(25);
    p_formLayout->addRow("Color:", p_colorButton);
    
    // Curve style combo box
    p_curveStyleBox = new QComboBox(this);
    p_curveStyleBox->addItem("No Curve", QVariant::fromValue(QwtPlotCurve::NoCurve));
    p_curveStyleBox->addItem("Line Plot", QVariant::fromValue(QwtPlotCurve::Lines));
    p_curveStyleBox->addItem("Stick Plot", QVariant::fromValue(QwtPlotCurve::Sticks));
    p_curveStyleBox->addItem("Step Plot", QVariant::fromValue(QwtPlotCurve::Steps));
    p_curveStyleBox->addItem("Scatter Dots", QVariant::fromValue(QwtPlotCurve::Dots));
    p_formLayout->addRow("Curve Type:", p_curveStyleBox);
    
    // Line thickness spin box
    p_thicknessBox = new QDoubleSpinBox(this);
    p_thicknessBox->setRange(0.0, 10.0);
    p_thicknessBox->setDecimals(1);
    p_thicknessBox->setSingleStep(0.5);
    p_formLayout->addRow("Line Width:", p_thicknessBox);
    
    // Line style combo box
    p_lineStyleBox = new QComboBox(this);
    p_lineStyleBox->addItem("None", QVariant::fromValue(Qt::NoPen));
    p_lineStyleBox->addItem(QString::fromUtf16(u"⸻ "), QVariant::fromValue(Qt::SolidLine));
    p_lineStyleBox->addItem("- - - ", QVariant::fromValue(Qt::DashLine));
    p_lineStyleBox->addItem(QString::fromUtf16(u"· · · "), QVariant::fromValue(Qt::DotLine));
    p_lineStyleBox->addItem(QString::fromUtf16(u"-·-·-"), QVariant::fromValue(Qt::DashDotLine));
    p_lineStyleBox->addItem(QString::fromUtf16(u"-··-··"), QVariant::fromValue(Qt::DashDotDotLine));
    p_formLayout->addRow("Line Style:", p_lineStyleBox);
    
    // Marker style combo box
    p_markerBox = new QComboBox(this);
    p_markerBox->addItem("None", QVariant::fromValue(QwtSymbol::NoSymbol));
    p_markerBox->addItem(QString::fromUtf16(u"●"), QVariant::fromValue(QwtSymbol::Ellipse));
    p_markerBox->addItem(QString::fromUtf16(u"■"), QVariant::fromValue(QwtSymbol::Rect));
    p_markerBox->addItem(QString::fromUtf16(u"⬥"), QVariant::fromValue(QwtSymbol::Diamond));
    p_markerBox->addItem(QString::fromUtf16(u"▲"), QVariant::fromValue(QwtSymbol::UTriangle));
    p_markerBox->addItem(QString::fromUtf16(u"▼"), QVariant::fromValue(QwtSymbol::DTriangle));
    p_markerBox->addItem(QString::fromUtf16(u"◀"), QVariant::fromValue(QwtSymbol::LTriangle));
    p_markerBox->addItem(QString::fromUtf16(u"▶"), QVariant::fromValue(QwtSymbol::RTriangle));
    p_markerBox->addItem(QString::fromUtf16(u"＋"), QVariant::fromValue(QwtSymbol::Cross));
    p_markerBox->addItem(QString::fromUtf16(u"⨯"), QVariant::fromValue(QwtSymbol::XCross));
    p_markerBox->addItem(QString::fromUtf16(u"—"), QVariant::fromValue(QwtSymbol::HLine));
    p_markerBox->addItem(QString::fromUtf16(u"︱"), QVariant::fromValue(QwtSymbol::VLine));
    p_markerBox->addItem(QString::fromUtf16(u"✳"), QVariant::fromValue(QwtSymbol::Star1));
    p_markerBox->addItem(QString::fromUtf16(u"✶"), QVariant::fromValue(QwtSymbol::Star2));
    p_markerBox->addItem(QString::fromUtf16(u"⬢"), QVariant::fromValue(QwtSymbol::Hexagon));
    p_formLayout->addRow("Marker:", p_markerBox);
    
    // Marker size spin box
    p_markerSizeBox = new QSpinBox(this);
    p_markerSizeBox->setRange(1, 20);
    p_formLayout->addRow("Marker Size:", p_markerSizeBox);
    
    // Visibility checkbox
    p_visibleBox = new QCheckBox(this);
    p_formLayout->addRow("Visible:", p_visibleBox);
    
    // Autoscale checkbox
    p_autoscaleBox = new QCheckBox(this);
    p_autoscaleBox->setToolTip("Controls whether the curve is included when calculating the axis limits for the autoscale operation");
    p_formLayout->addRow("Autoscale?", p_autoscaleBox);
    
    // Y-axis combo box
    p_yAxisBox = new QComboBox(this);
    p_yAxisBox->addItem("Left", QVariant::fromValue(QwtAxis::YLeft));
    p_yAxisBox->addItem("Right", QVariant::fromValue(QwtAxis::YRight));
    p_formLayout->addRow("Y Axis:", p_yAxisBox);
    
    // Configure label alignment as in the original implementation
    for (int i = 0; i < p_formLayout->rowCount(); ++i) {
        QLayoutItem *item = p_formLayout->itemAt(i, QFormLayout::LabelRole);
        if (item && item->widget()) {
            auto lbl = qobject_cast<QLabel*>(item->widget());
            if (lbl) {
                lbl->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
                lbl->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Expanding);
            }
        }
    }
}

void CurveAppearanceWidget::setupConnections()
{
    // Preset connections
    connect(p_presetBox, qOverload<int>(&QComboBox::currentIndexChanged), this, &CurveAppearanceWidget::onPresetSelected);
    connect(p_savePresetButton, &QPushButton::clicked, this, &CurveAppearanceWidget::onSavePresetClicked);
    connect(p_deletePresetButton, &QPushButton::clicked, this, &CurveAppearanceWidget::onDeletePresetClicked);
    
    // Appearance control connections
    connect(p_colorButton, &QPushButton::clicked, this, &CurveAppearanceWidget::onColorButtonClicked);
    connect(p_curveStyleBox, qOverload<int>(&QComboBox::currentIndexChanged), this, &CurveAppearanceWidget::onCurveStyleChanged);
    connect(p_thicknessBox, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &CurveAppearanceWidget::onLineThicknessChanged);
    connect(p_lineStyleBox, qOverload<int>(&QComboBox::currentIndexChanged), this, &CurveAppearanceWidget::onLineStyleChanged);
    connect(p_markerBox, qOverload<int>(&QComboBox::currentIndexChanged), this, &CurveAppearanceWidget::onMarkerStyleChanged);
    connect(p_markerSizeBox, qOverload<int>(&QSpinBox::valueChanged), this, &CurveAppearanceWidget::onMarkerSizeChanged);
    connect(p_visibleBox, &QCheckBox::toggled, this, &CurveAppearanceWidget::onVisibilityChanged);
    connect(p_autoscaleBox, &QCheckBox::toggled, this, &CurveAppearanceWidget::onAutoscaleChanged);
    connect(p_yAxisBox, qOverload<int>(&QComboBox::currentIndexChanged), this, &CurveAppearanceWidget::onYAxisChanged);
}

void CurveAppearanceWidget::initializeFromCurve(BlackchirpPlotCurveBase *curve)
{
    if (!curve) {
        return;
    }
    
    d_blockSignals = true;
    
    // Read current curve properties
    d_currentAppearance.color = curve->pen().color();
    d_currentAppearance.curveStyle = curve->style();
    d_currentAppearance.lineThickness = curve->pen().widthF();
    d_currentAppearance.lineStyle = curve->pen().style();
    d_currentAppearance.markerStyle = curve->symbol() ? curve->symbol()->style() : QwtSymbol::NoSymbol;
    d_currentAppearance.markerSize = curve->symbol() ? curve->symbol()->size().width() : 7;
    d_currentAppearance.visible = curve->isVisible();
    d_currentAppearance.autoscale = curve->testItemAttribute(QwtPlotItem::AutoScale);
    d_currentAppearance.yAxis = curve->yAxis();
    
    // Update UI to reflect current values
    setCurrentAppearance(d_currentAppearance);
    
    d_blockSignals = false;
}

void CurveAppearanceWidget::applyToCurve(BlackchirpPlotCurveBase *curve)
{
    if (!curve) {
        return;
    }
    
    curve->setColor(d_currentAppearance.color);
    curve->setCurveStyle(d_currentAppearance.curveStyle);
    curve->setLineThickness(d_currentAppearance.lineThickness);
    curve->setLineStyle(d_currentAppearance.lineStyle);
    curve->setMarkerStyle(d_currentAppearance.markerStyle);
    curve->setMarkerSize(d_currentAppearance.markerSize);
    curve->setCurveVisible(d_currentAppearance.visible);
    curve->setCurveAutoscale(d_currentAppearance.autoscale);
    curve->setCurveAxisY(static_cast<QwtPlot::Axis>(d_currentAppearance.yAxis)); // TODO: Remove cast when BlackchirpPlotCurveBase migrates to QwtAxisId
}

CurveAppearanceWidget::CurveAppearance CurveAppearanceWidget::getCurrentAppearance() const
{
    return d_currentAppearance;
}

void CurveAppearanceWidget::setCurrentAppearance(const CurveAppearance &appearance)
{
    d_blockSignals = true;
    
    d_currentAppearance = appearance;
    
    // Update color button
    QColor borderColor = palette().color(QPalette::Text);
    QColor textColor = (appearance.color.lightness() > 128) ? Qt::black : Qt::white;
    QString colorStyle = QString("background-color: %1; border: 1px solid %2; color: %3;")
                        .arg(appearance.color.name())
                        .arg(borderColor.name())
                        .arg(textColor.name());
    p_colorButton->setStyleSheet(colorStyle);
    
    // Update combo boxes
    p_curveStyleBox->setCurrentIndex(p_curveStyleBox->findData(QVariant::fromValue(appearance.curveStyle)));
    p_lineStyleBox->setCurrentIndex(p_lineStyleBox->findData(QVariant::fromValue(appearance.lineStyle)));
    p_markerBox->setCurrentIndex(p_markerBox->findData(QVariant::fromValue(appearance.markerStyle)));
    p_yAxisBox->setCurrentIndex(p_yAxisBox->findData(QVariant::fromValue(appearance.yAxis)));
    
    // Update spin boxes
    p_thicknessBox->setValue(appearance.lineThickness);
    p_markerSizeBox->setValue(appearance.markerSize);
    
    // Update checkboxes
    p_visibleBox->setChecked(appearance.visible);
    p_autoscaleBox->setChecked(appearance.autoscale);
    
    d_blockSignals = false;
}

void CurveAppearanceWidget::setColorButtonEnabled(bool enabled)
{
    p_colorButton->setEnabled(enabled);
}

void CurveAppearanceWidget::setYAxisControlEnabled(bool enabled)
{
    p_yAxisBox->setEnabled(enabled);
    // Update label too
    for (int i = 0; i < p_formLayout->rowCount(); ++i) {
        if (p_formLayout->itemAt(i, QFormLayout::FieldRole)->widget() == p_yAxisBox) {
            auto lbl = qobject_cast<QLabel*>(p_formLayout->itemAt(i, QFormLayout::LabelRole)->widget());
            if (lbl) {
                lbl->setEnabled(enabled);
            }
            break;
        }
    }
}

void CurveAppearanceWidget::updateColorDisplay(const QColor &color)
{
    d_currentAppearance.color = color;
    QColor borderColor = palette().color(QPalette::Text);
    QColor textColor = (color.lightness() > 128) ? Qt::black : Qt::white;
    QString colorStyle = QString("background-color: %1; border: 1px solid %2; color: %3;")
                        .arg(color.name())
                        .arg(borderColor.name())
                        .arg(textColor.name());
    p_colorButton->setStyleSheet(colorStyle);
    emitAppearanceChanged();
}

void CurveAppearanceWidget::onColorButtonClicked()
{
    // Emit signal to let parent handle color dialog
    // This works better when embedded in context menus
    emit colorChangeRequested();
}

void CurveAppearanceWidget::onCurveStyleChanged(int index)
{
    if (d_blockSignals) return;
    
    auto newStyle = p_curveStyleBox->itemData(index).value<QwtPlotCurve::CurveStyle>();
    if (newStyle != d_currentAppearance.curveStyle) {
        d_currentAppearance.curveStyle = newStyle;
        emitAppearanceChanged();
    }
}

void CurveAppearanceWidget::onLineThicknessChanged(double value)
{
    if (d_blockSignals) return;
    
    if (value != d_currentAppearance.lineThickness) {
        d_currentAppearance.lineThickness = value;
        emitAppearanceChanged();
    }
}

void CurveAppearanceWidget::onLineStyleChanged(int index)
{
    if (d_blockSignals) return;
    
    auto newStyle = p_lineStyleBox->itemData(index).value<Qt::PenStyle>();
    if (newStyle != d_currentAppearance.lineStyle) {
        d_currentAppearance.lineStyle = newStyle;
        emitAppearanceChanged();
    }
}

void CurveAppearanceWidget::onMarkerStyleChanged(int index)
{
    if (d_blockSignals) return;
    
    auto newStyle = p_markerBox->itemData(index).value<QwtSymbol::Style>();
    if (newStyle != d_currentAppearance.markerStyle) {
        d_currentAppearance.markerStyle = newStyle;
        emitAppearanceChanged();
    }
}

void CurveAppearanceWidget::onMarkerSizeChanged(int value)
{
    if (d_blockSignals) return;
    
    if (value != d_currentAppearance.markerSize) {
        d_currentAppearance.markerSize = value;
        emitAppearanceChanged();
    }
}

void CurveAppearanceWidget::onVisibilityChanged(bool visible)
{
    if (d_blockSignals) return;
    
    if (visible != d_currentAppearance.visible) {
        d_currentAppearance.visible = visible;
        emitAppearanceChanged();
    }
}

void CurveAppearanceWidget::onAutoscaleChanged(bool enabled)
{
    if (d_blockSignals) return;
    
    if (enabled != d_currentAppearance.autoscale) {
        d_currentAppearance.autoscale = enabled;
        emitAppearanceChanged();
    }
}

void CurveAppearanceWidget::onYAxisChanged(int index)
{
    if (d_blockSignals) return;
    
    auto newAxis = p_yAxisBox->itemData(index).value<QwtAxisId>();
    if (newAxis != d_currentAppearance.yAxis) {
        d_currentAppearance.yAxis = newAxis;
        emitAppearanceChanged();
    }
}

void CurveAppearanceWidget::emitAppearanceChanged()
{
    if (!d_blockSignals) {
        emit curveAppearanceChanged(d_currentAppearance);
    }
}

void CurveAppearanceWidget::initializeFromOverlay(std::shared_ptr<OverlayBase> overlay)
{
    if (!overlay) {
        return;
    }
    
    // Create appearance structure from overlay metadata
    CurveAppearance appearance;
    
    // Load color (default to palette text color if not set)
    QVariant colorVar = overlay->getCurveMetadata(BC::Key::bcCurveColor);
    if (colorVar.isValid()) {
        appearance.color = colorVar.value<QColor>();
    } else {
        appearance.color = palette().color(QPalette::Text);
    }
    
    // Load curve style (default to Lines)
    QVariant curveStyleVar = overlay->getCurveMetadata(BC::Key::bcCurveCurveStyle);
    if (curveStyleVar.isValid()) {
        appearance.curveStyle = static_cast<QwtPlotCurve::CurveStyle>(curveStyleVar.toInt());
    } else {
        appearance.curveStyle = QwtPlotCurve::Lines;
    }
    
    // Load line thickness (default to 1.0)
    QVariant thicknessVar = overlay->getCurveMetadata(BC::Key::bcCurveThickness);
    if (thicknessVar.isValid()) {
        appearance.lineThickness = thicknessVar.toDouble();
    } else {
        appearance.lineThickness = 1.0;
    }
    
    // Load line style (default to SolidLine)
    QVariant lineStyleVar = overlay->getCurveMetadata(BC::Key::bcCurveLineStyle);
    if (lineStyleVar.isValid()) {
        appearance.lineStyle = static_cast<Qt::PenStyle>(lineStyleVar.toInt());
    } else {
        appearance.lineStyle = Qt::SolidLine;
    }
    
    // Load marker style (default to NoSymbol)
    QVariant markerVar = overlay->getCurveMetadata(BC::Key::bcCurveMarker);
    if (markerVar.isValid()) {
        appearance.markerStyle = static_cast<QwtSymbol::Style>(markerVar.toInt());
    } else {
        appearance.markerStyle = QwtSymbol::NoSymbol;
    }
    
    // Load marker size (default to 7)
    QVariant markerSizeVar = overlay->getCurveMetadata(BC::Key::bcCurveMarkerSize);
    if (markerSizeVar.isValid()) {
        appearance.markerSize = markerSizeVar.toInt();
    } else {
        appearance.markerSize = 7;
    }
    
    // Load visibility (default to true)
    QVariant visibleVar = overlay->getCurveMetadata(BC::Key::bcCurveVisible);
    if (visibleVar.isValid()) {
        appearance.visible = visibleVar.toBool();
    } else {
        appearance.visible = true;
    }
    
    // Load autoscale (default to true)
    QVariant autoscaleVar = overlay->getCurveMetadata(BC::Key::bcCurveAutoscale);
    if (autoscaleVar.isValid()) {
        appearance.autoscale = autoscaleVar.toBool();
    } else {
        appearance.autoscale = true;
    }
    
    // Load Y axis (default to YLeft)
    QVariant yAxisVar = overlay->getCurveMetadata(BC::Key::bcCurveAxisY);
    if (yAxisVar.isValid()) {
        // Convert from old QwtPlot::Axis to new QwtAxisId
        QwtPlot::Axis oldAxis = static_cast<QwtPlot::Axis>(yAxisVar.toInt());
        switch (oldAxis) {
            case QwtPlot::yLeft:
                appearance.yAxis = QwtAxis::YLeft;
                break;
            case QwtPlot::yRight:
                appearance.yAxis = QwtAxis::YRight;
                break;
            default:
                appearance.yAxis = QwtAxis::YLeft;
                break;
        }
    } else {
        appearance.yAxis = QwtAxis::YLeft;
    }
    
    // Apply appearance to widget (this will handle signal blocking internally)
    setCurrentAppearance(appearance);
}

void CurveAppearanceWidget::applyToOverlay(std::shared_ptr<OverlayBase> overlay)
{
    if (!overlay) {
        return;
    }
    
    // Save all appearance properties to overlay metadata
    overlay->setCurveMetadata(BC::Key::bcCurveColor, d_currentAppearance.color);
    overlay->setCurveMetadata(BC::Key::bcCurveCurveStyle, static_cast<int>(d_currentAppearance.curveStyle));
    overlay->setCurveMetadata(BC::Key::bcCurveThickness, d_currentAppearance.lineThickness);
    overlay->setCurveMetadata(BC::Key::bcCurveLineStyle, static_cast<int>(d_currentAppearance.lineStyle));
    overlay->setCurveMetadata(BC::Key::bcCurveMarker, static_cast<int>(d_currentAppearance.markerStyle));
    overlay->setCurveMetadata(BC::Key::bcCurveMarkerSize, d_currentAppearance.markerSize);
    overlay->setCurveMetadata(BC::Key::bcCurveVisible, d_currentAppearance.visible);
    overlay->setCurveMetadata(BC::Key::bcCurveAutoscale, d_currentAppearance.autoscale);
    
    // Convert QwtAxisId back to QwtPlot::Axis for storage
    QwtPlot::Axis oldAxis;
    switch (d_currentAppearance.yAxis) {
        case QwtAxis::YLeft:
            oldAxis = QwtPlot::yLeft;
            break;
        case QwtAxis::YRight:
            oldAxis = QwtPlot::yRight;
            break;
        default:
            oldAxis = QwtPlot::yLeft;
            break;
    }
    overlay->setCurveMetadata(BC::Key::bcCurveAxisY, static_cast<int>(oldAxis));
}

// === PRESET MANAGEMENT METHODS ===

void CurveAppearanceWidget::setPresetManager(CurveAppearancePresetManager *manager)
{
    p_presetManager = manager;
    refreshPresetList();
}

void CurveAppearanceWidget::refreshPresetList()
{
    if (!p_presetManager) {
        return;
    }
    
    d_blockSignals = true;
    
    p_presetBox->clear();
    p_presetBox->addItem("Select preset...", QString()); // Empty string for no selection
    
    QStringList presetNames = p_presetManager->getPresetNames();
    for (const QString &name : presetNames) {
        p_presetBox->addItem(name, name);
    }
    
    d_blockSignals = false;
    
    // Update delete button state
    updateDeleteButtonState();
}

void CurveAppearanceWidget::applyPreset(const QString &presetName)
{
    if (!p_presetManager || presetName.isEmpty()) {
        return;
    }
    
    if (!p_presetManager->hasPreset(presetName)) {
        qWarning() << "Preset not found:" << presetName;
        return;
    }
    
    auto preset = p_presetManager->getPreset(presetName);
    setCurrentAppearance(preset.appearance);
    
    // Manually emit signal to trigger visual updates since setCurrentAppearance blocks signals
    emit curveAppearanceChanged(d_currentAppearance);
    
    // Mark preset as used
    p_presetManager->markPresetUsed(presetName);
    
    qDebug() << "Applied preset:" << presetName;
}

void CurveAppearanceWidget::saveCurrentAsPreset(const QString &presetName)
{
    if (!p_presetManager || presetName.isEmpty()) {
        return;
    }
    
    bool success = p_presetManager->savePreset(presetName, d_currentAppearance);
    if (success) {
        refreshPresetList();
        
        // Select the newly saved preset
        int index = p_presetBox->findData(presetName);
        if (index >= 0) {
            d_blockSignals = true;
            p_presetBox->setCurrentIndex(index);
            d_blockSignals = false;
        }
        
        qDebug() << "Saved preset:" << presetName;
    } else {
        QMessageBox::warning(this, "Save Failed", 
                           QString("Failed to save preset '%1'. Please check the name is valid.").arg(presetName));
    }
}

void CurveAppearanceWidget::deletePreset(const QString &presetName)
{
    if (!p_presetManager || presetName.isEmpty()) {
        return;
    }
    
    bool success = p_presetManager->deletePreset(presetName);
    if (success) {
        refreshPresetList();
        // Reset to "Select preset..."
        p_presetBox->setCurrentIndex(0);
        qDebug() << "Deleted preset:" << presetName;
    } else {
        QMessageBox::warning(this, "Delete Failed", 
                           QString("Failed to delete preset '%1'. Default presets cannot be deleted.").arg(presetName));
    }
}

void CurveAppearanceWidget::onPresetSelected(int index)
{
    if (d_blockSignals || !p_presetManager) {
        return;
    }
    
    QString presetName = p_presetBox->itemData(index).toString();
    if (!presetName.isEmpty()) {
        applyPreset(presetName);
    }
    
    updateDeleteButtonState();
}

void CurveAppearanceWidget::onSavePresetClicked()
{
    if (!p_presetManager) {
        return;
    }
    
    // Suggest a name based on current settings and emit signal for external handling
    QString suggestedName = generatePresetSuggestion();
    emit presetSaveRequested(suggestedName);
}

void CurveAppearanceWidget::onDeletePresetClicked()
{
    if (!p_presetManager) {
        return;
    }
    
    QString currentPreset = p_presetBox->currentData().toString();
    if (currentPreset.isEmpty()) {
        return;
    }
    
    // Emit signal for external handling of delete confirmation dialog
    emit presetDeleteRequested(currentPreset);
}

void CurveAppearanceWidget::updateDeleteButtonState()
{
    if (!p_presetManager) {
        p_deletePresetButton->setEnabled(false);
        return;
    }
    
    QString currentPreset = p_presetBox->currentData().toString();
    if (currentPreset.isEmpty()) {
        p_deletePresetButton->setEnabled(false);
        return;
    }
    
    auto preset = p_presetManager->getPreset(currentPreset);
    p_deletePresetButton->setEnabled(!preset.isDefault);
}

QString CurveAppearanceWidget::generatePresetSuggestion() const
{
    // Generate a suggested name based on current appearance
    QString suggestion = "Custom";
    
    // Add curve type to suggestion
    if (d_currentAppearance.curveStyle == QwtPlotCurve::Lines) {
        suggestion = "Curve";
    } else if (d_currentAppearance.curveStyle == QwtPlotCurve::Sticks) {
        suggestion = "Stem";
    } else if (d_currentAppearance.curveStyle == QwtPlotCurve::NoCurve) {
        suggestion = "Scatter";
    }
    
    // Add color info if it's a common color
    QColor color = d_currentAppearance.color;
    if (color == Qt::red) {
        suggestion += " - Red";
    } else if (color == Qt::blue) {
        suggestion += " - Blue";
    } else if (color == Qt::green || color == Qt::darkGreen) {
        suggestion += " - Green";
    } else if (color == Qt::black) {
        suggestion += " - Black";
    }
    
    return suggestion;
}