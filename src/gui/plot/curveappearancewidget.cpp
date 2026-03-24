#include "curveappearancewidget.h"
#include "curveappearancepresetmanager.h"
#include "blackchirpplotcurve.h"
#include <data/experiment/overlaybase.h>

#include <QColorDialog>
#include <QLabel>
#include <QInputDialog>
#include <QMessageBox>
#include <QHBoxLayout>
#include <cmath>

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
    // Create main vertical layout with compact spacing
    auto mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(6, 6, 6, 6);
    mainLayout->setSpacing(4);
    
    // === PRESET CONTROLS GROUP ===
    auto presetGroup = new QGroupBox("Presets", this);
    presetGroup->setFlat(true);
    auto presetLayout = new QGridLayout(presetGroup);
    presetLayout->setContentsMargins(3, 2, 3, 3);
    presetLayout->setSpacing(3);
    
    // Preset selection (full width)
    p_presetBox = new QComboBox(presetGroup);
    p_presetBox->setToolTip("Select a preset to apply or create a new preset");
    auto presetLabel = new QLabel("Preset:", presetGroup);
    presetLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    presetLayout->addWidget(presetLabel, 0, 0);
    presetLayout->addWidget(p_presetBox, 0, 1, 1, 2);
    
    // Preset action buttons (compact, side by side)
    p_savePresetButton = new QPushButton("Save", presetGroup);
    p_savePresetButton->setToolTip("Save current appearance as a new preset");
    p_savePresetButton->setMaximumWidth(60);
    
    p_deletePresetButton = new QPushButton("Delete", presetGroup);
    p_deletePresetButton->setToolTip("Delete the selected preset");
    p_deletePresetButton->setMaximumWidth(60);
    p_deletePresetButton->setEnabled(false);
    
    presetLayout->addWidget(p_savePresetButton, 1, 1);
    presetLayout->addWidget(p_deletePresetButton, 1, 2);
    
    mainLayout->addWidget(presetGroup);
    
    // === APPEARANCE CONTROLS GROUP ===
    auto appearanceGroup = new QGroupBox("Appearance", this);
    appearanceGroup->setFlat(true);
    auto appearanceLayout = new QVBoxLayout(appearanceGroup);
    appearanceLayout->setContentsMargins(3, 2, 3, 3);
    appearanceLayout->setSpacing(3);
    
    // Color and curve type row
    auto colorCurveLayout = new QGridLayout();
    colorCurveLayout->setSpacing(3);
    
    p_colorButton = new QPushButton("Choose Color...", appearanceGroup);
    p_colorButton->setMinimumHeight(25);
    auto colorLabel = new QLabel("Color:", appearanceGroup);
    colorLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    colorCurveLayout->addWidget(colorLabel, 0, 0);
    colorCurveLayout->addWidget(p_colorButton, 0, 1);
    
    p_curveStyleBox = new QComboBox(appearanceGroup);
    p_curveStyleBox->addItem("No Curve", QVariant::fromValue(QwtPlotCurve::NoCurve));
    p_curveStyleBox->addItem("Line Plot", QVariant::fromValue(QwtPlotCurve::Lines));
    p_curveStyleBox->addItem("Stick Plot", QVariant::fromValue(QwtPlotCurve::Sticks));
    p_curveStyleBox->addItem("Step Plot", QVariant::fromValue(QwtPlotCurve::Steps));
    p_curveStyleBox->addItem("Scatter Dots", QVariant::fromValue(QwtPlotCurve::Dots));
    auto typeLabel = new QLabel("Type:", appearanceGroup);
    typeLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    colorCurveLayout->addWidget(typeLabel, 0, 2);
    colorCurveLayout->addWidget(p_curveStyleBox, 0, 3);
    
    appearanceLayout->addLayout(colorCurveLayout);
    
    // Line style and thickness row  
    auto lineLayout = new QGridLayout();
    lineLayout->setSpacing(3);
    
    p_thicknessBox = new QDoubleSpinBox(appearanceGroup);
    p_thicknessBox->setRange(0.0, 10.0);
    p_thicknessBox->setDecimals(1);
    p_thicknessBox->setSingleStep(0.5);
    p_thicknessBox->setMaximumWidth(60);
    auto widthLabel = new QLabel("Width:", appearanceGroup);
    widthLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    lineLayout->addWidget(widthLabel, 0, 0);
    lineLayout->addWidget(p_thicknessBox, 0, 1);
    
    p_lineStyleBox = new QComboBox(appearanceGroup);
    p_lineStyleBox->addItem("None", QVariant::fromValue(Qt::NoPen));
    p_lineStyleBox->addItem(QString::fromUtf16(u"⸻ "), QVariant::fromValue(Qt::SolidLine));
    p_lineStyleBox->addItem("- - - ", QVariant::fromValue(Qt::DashLine));
    p_lineStyleBox->addItem(QString::fromUtf16(u"· · · "), QVariant::fromValue(Qt::DotLine));
    p_lineStyleBox->addItem(QString::fromUtf16(u"-·-·-"), QVariant::fromValue(Qt::DashDotLine));
    p_lineStyleBox->addItem(QString::fromUtf16(u"-··-··"), QVariant::fromValue(Qt::DashDotDotLine));
    auto styleLabel = new QLabel("Style:", appearanceGroup);
    styleLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    lineLayout->addWidget(styleLabel, 0, 2);
    lineLayout->addWidget(p_lineStyleBox, 0, 3);
    
    appearanceLayout->addLayout(lineLayout);
    
    // Marker style and size row
    auto markerLayout = new QGridLayout();
    markerLayout->setSpacing(3);
    
    p_markerBox = new QComboBox(appearanceGroup);
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
    auto markerLabel = new QLabel("Marker:", appearanceGroup);
    markerLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    markerLayout->addWidget(markerLabel, 0, 0);
    markerLayout->addWidget(p_markerBox, 0, 1);
    
    p_markerSizeBox = new QSpinBox(appearanceGroup);
    p_markerSizeBox->setRange(1, 20);
    p_markerSizeBox->setMaximumWidth(50);
    auto sizeLabel = new QLabel("Size:", appearanceGroup);
    sizeLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    markerLayout->addWidget(sizeLabel, 0, 2);
    markerLayout->addWidget(p_markerSizeBox, 0, 3);
    
    appearanceLayout->addLayout(markerLayout);
    
    // Options row (checkboxes and Y-axis)
    auto optionsLayout = new QGridLayout();
    optionsLayout->setSpacing(3);
    
    p_visibleBox = new QCheckBox("Visible", appearanceGroup);
    optionsLayout->addWidget(p_visibleBox, 0, 0);
    
    p_autoscaleBox = new QCheckBox("Autoscale", appearanceGroup);
    p_autoscaleBox->setToolTip("Controls whether the curve is included when calculating the axis limits for the autoscale operation");
    optionsLayout->addWidget(p_autoscaleBox, 0, 1);
    
    p_yAxisBox = new QComboBox(appearanceGroup);
    p_yAxisBox->addItem("Left", QVariant::fromValue(QwtAxis::YLeft));
    p_yAxisBox->addItem("Right", QVariant::fromValue(QwtAxis::YRight));
    auto yAxisLabel = new QLabel("Y Axis:", appearanceGroup);
    yAxisLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    optionsLayout->addWidget(yAxisLabel, 0, 2);
    optionsLayout->addWidget(p_yAxisBox, 0, 3);
    
    appearanceLayout->addLayout(optionsLayout);
    
    mainLayout->addWidget(appearanceGroup);
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
    // Find and update the Y Axis label - it's the sibling widget in the same layout
    QWidget *parent = p_yAxisBox->parentWidget();
    if (parent) {
        auto labels = parent->findChildren<QLabel*>();
        for (QLabel *label : labels) {
            if (label->text() == "Y Axis:") {
                label->setEnabled(enabled);
                break;
            }
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

    // Overlay curve metadata stores QWT enum int values (matching what
    // BlackchirpPlotCurveBase uses via OverlayMetadataStorage). Read them
    // directly as QWT types.
    CurveAppearance appearance;
    
    // Load color (default to palette text color if not set)
    QVariant oldColorVar = overlay->getCurveMetadata(BC::Key::bcCurveColor);
    if (oldColorVar.isValid()) {
        appearance.color = oldColorVar.value<QColor>();
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

    // Store QWT enum int values directly. The curve rendering
    // (BlackchirpPlotCurveBase::configureSymbol etc.) reads these via
    // OverlayMetadataStorage and interprets them as QWT enums.
    // Do NOT go through BC::Data conversion — BC::Data::MarkerStyle starts
    // at 0 for NoSymbol while QwtSymbol::Style starts at -1, which causes
    // an off-by-one when the curve reads the stored int.
    overlay->setCurveMetadata(BC::Key::bcCurveColor, d_currentAppearance.color);
    overlay->setCurveMetadata(BC::Key::bcCurveCurveStyle, static_cast<int>(d_currentAppearance.curveStyle));
    overlay->setCurveMetadata(BC::Key::bcCurveThickness, d_currentAppearance.lineThickness);
    overlay->setCurveMetadata(BC::Key::bcCurveLineStyle, static_cast<int>(d_currentAppearance.lineStyle));
    overlay->setCurveMetadata(BC::Key::bcCurveMarker, static_cast<int>(d_currentAppearance.markerStyle));
    overlay->setCurveMetadata(BC::Key::bcCurveMarkerSize, d_currentAppearance.markerSize);
    overlay->setCurveMetadata(BC::Key::bcCurveVisible, d_currentAppearance.visible);
    overlay->setCurveMetadata(BC::Key::bcCurveAutoscale, d_currentAppearance.autoscale);
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
    QString suggestion;
    
    // Start with curve style
    if (d_currentAppearance.curveStyle == QwtPlotCurve::Lines) {
        suggestion = "Curve";
    } else if (d_currentAppearance.curveStyle == QwtPlotCurve::Sticks) {
        suggestion = "Stem";
    } else if (d_currentAppearance.curveStyle == QwtPlotCurve::NoCurve) {
        suggestion = "Scatter";
    } else {
        suggestion = "Custom";
    }
    
    // Add line thickness and style for non-NoCurve styles
    if (d_currentAppearance.curveStyle != QwtPlotCurve::NoCurve) {
        double width = d_currentAppearance.lineThickness;
        if (width < 1.5) {
            suggestion += " Thin";
        } else if (width <= 3.0) {
            suggestion += " Medium";
        } else if (width <= 5.0) {
            suggestion += " Thick";
        } else {
            suggestion += " VeryThick";
        }
        
        // Add line style
        QString lineStyle = getLineStyleName(d_currentAppearance.lineStyle);
        if (lineStyle != "Solid") { // Only add if not solid (default)
            suggestion += " " + lineStyle;
        }
    }
    
    // Add marker information if markers are present
    if (d_currentAppearance.markerStyle != QwtSymbol::NoSymbol) {
        // Add marker shape name
        QString markerName = getMarkerShapeName(d_currentAppearance.markerStyle);
        suggestion += " " + markerName;
        
        // Add marker size category
        double size = d_currentAppearance.markerSize;
        if (size < 3) {
            suggestion += " Small";
        } else if (size <= 6) {
            suggestion += " Medium";
        } else if (size <= 9) {
            suggestion += " Large";
        } else {
            suggestion += " VeryLarge";
        }
    }
    
    // Add color description
    QString colorDesc = getColorDescription(d_currentAppearance.color);
    suggestion += " " + colorDesc;
    
    return suggestion;
}

QString CurveAppearanceWidget::getMarkerShapeName(QwtSymbol::Style style) const
{
    switch (style) {
        case QwtSymbol::Ellipse:
            return "Circle";
        case QwtSymbol::Rect:
            return "Square";
        case QwtSymbol::Diamond:
            return "Diamond";
        case QwtSymbol::Triangle:
            return "Triangle";
        case QwtSymbol::DTriangle:
            return "DownTriangle";
        case QwtSymbol::UTriangle:
            return "UpTriangle";
        case QwtSymbol::LTriangle:
            return "LeftTriangle";
        case QwtSymbol::RTriangle:
            return "RightTriangle";
        case QwtSymbol::Cross:
            return "Cross";
        case QwtSymbol::XCross:
            return "XCross";
        case QwtSymbol::HLine:
            return "HLine";
        case QwtSymbol::VLine:
            return "VLine";
        case QwtSymbol::Star1:
            return "Star";
        case QwtSymbol::Star2:
            return "Star6";
        case QwtSymbol::Hexagon:
            return "Hexagon";
        default:
            return "Marker";
    }
}

QString CurveAppearanceWidget::getLineStyleName(Qt::PenStyle style) const
{
    switch (style) {
        case Qt::SolidLine:
            return "Solid";
        case Qt::DashLine:
            return "Dashed";
        case Qt::DotLine:
            return "Dotted";
        case Qt::DashDotLine:
            return "DashDot";
        case Qt::DashDotDotLine:
            return "DashDotDot";
        default:
            return "Solid";
    }
}

QString CurveAppearanceWidget::getColorDescription(const QColor &color) const
{
    int r = color.red();
    int g = color.green();
    int b = color.blue();
    double lightness = color.lightnessF(); // 0.0 to 1.0
    
    // Check if it's grayscale (RGB values within 10 units)
    if (std::abs(r - g) <= 10 && std::abs(g - b) <= 10 && std::abs(r - b) <= 10) {
        if (lightness < 0.1) {
            return "Black";
        } else if (lightness > 0.9) {
            return "White";
        } else {
            // Add light/dark modifier for gray
            QString grayName = "Gray";
            if (lightness >= 0.75) {
                grayName = "Light " + grayName;
            } else if (lightness <= 0.25) {
                grayName = "Dark " + grayName;
            }
            return grayName;
        }
    }
    
    // Determine base color from RGB values
    QString baseColor;
    
    // Can use these later if more granular labels are desired.
    // Find the dominant color component
    // int maxComponent = std::max({r, g, b});
    // int minComponent = std::min({r, g, b});
    
    // Calculate color ratios for better classification
    // double rRatio = static_cast<double>(r) / 255.0;
    // double gRatio = static_cast<double>(g) / 255.0;
    // double bRatio = static_cast<double>(b) / 255.0;
    
    if (r >= g && r >= b) {
        // Red is dominant
        if (g > b * 1.5) {
            // Significant green component
            if (g >= r * 0.8) {
                baseColor = "Yellow";
            } else if (g >= r * 0.5) {
                baseColor = "Orange";
            } else {
                baseColor = "Red";
            }
        } else if (b > g * 1.2) {
            // Some blue component
            if (b >= r * 0.6) {
                baseColor = "Purple";
            } else {
                baseColor = "Pink";
            }
        } else {
            // Check for brown (low saturation red with some green)
            if (lightness < 0.6 && g >= r * 0.3 && g < r * 0.8) {
                baseColor = "Brown";
            } else {
                baseColor = "Red";
            }
        }
    } else if (g >= r && g >= b) {
        // Green is dominant
        if (r > b * 1.2) {
            // Some red component
            if (r >= g * 0.8) {
                baseColor = "Yellow";
            } else {
                baseColor = "Green";
            }
        } else if (b > r * 1.2) {
            // Some blue component
            baseColor = "Green";
        } else {
            baseColor = "Green";
        }
    } else {
        // Blue is dominant
        if (r > g * 1.2) {
            // Some red component
            if (r >= b * 0.6) {
                baseColor = "Purple";
            } else {
                baseColor = "Blue";
            }
        } else if (g > r * 1.2) {
            // Some green component
            baseColor = "Blue";
        } else {
            baseColor = "Blue";
        }
    }
    
    // Add lightness modifiers
    if (lightness >= 0.75 && lightness < 0.9) {
        baseColor = "Light " + baseColor;
    } else if (lightness <= 0.25 && lightness > 0.1) {
        baseColor = "Dark " + baseColor;
    }
    
    return baseColor;
}
