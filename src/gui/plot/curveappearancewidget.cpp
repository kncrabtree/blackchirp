#include "curveappearancewidget.h"
#include "blackchirpplotcurve.h"

#include <QColorDialog>
#include <QLabel>

CurveAppearanceWidget::CurveAppearanceWidget(QWidget *parent)
    : QWidget(parent), d_blockSignals(false)
{
    setupUI();
    setupConnections();
    
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
        auto lbl = qobject_cast<QLabel*>(p_formLayout->itemAt(i, QFormLayout::LabelRole)->widget());
        if (lbl) {
            lbl->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
            lbl->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Expanding);
        }
    }
}

void CurveAppearanceWidget::setupConnections()
{
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