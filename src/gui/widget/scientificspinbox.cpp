#include "scientificspinbox.h"

#include <QApplication>
#include <QClipboard>
#include <QContextMenuEvent>
#include <QFontMetrics>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMenu>
#include <QActionGroup>
#include <QSpinBox>
#include <QWidgetAction>
#include <QWheelEvent>
#include <QRegularExpression>
#include <QDebug>
#include <charconv>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <limits>

#include <gui/style/themecolors.h>
#include <gui/util/numericformat.h>

using namespace Qt::Literals::StringLiterals;

ScientificSpinBox::ScientificSpinBox(QWidget *parent)
    : QAbstractSpinBox(parent)
{
    lineEdit()->setMaxLength(MAX_INPUT_LENGTH);

    connect(lineEdit(), &QLineEdit::editingFinished, this, &ScientificSpinBox::onEditingFinished);
    connect(lineEdit(), &QLineEdit::textChanged, this, &ScientificSpinBox::onTextChanged);

    updateDisplayText();
    lineEdit()->setText(d_cachedDisplayText);

    setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
}

double ScientificSpinBox::value() const
{
    return d_value;
}

void ScientificSpinBox::setValue(double value)
{
    value = std::clamp(value, d_minimum, d_maximum);

    if (qFuzzyCompare(d_value, value))
        return;

    d_value = value;

    if (!d_isEditing && !d_precisionExplicit)
        d_displayPrecision = -1;

    updateDisplayText();
    updateEditText();

    if (!d_isEditing)
        lineEdit()->setText(d_cachedDisplayText);

    emitValueChanged();
}

double ScientificSpinBox::minimum() const
{
    return d_minimum;
}

void ScientificSpinBox::setMinimum(double min)
{
    d_minimum = min;
    if (d_value < d_minimum)
        setValue(d_minimum);
}

double ScientificSpinBox::maximum() const
{
    return d_maximum;
}

void ScientificSpinBox::setMaximum(double max)
{
    d_maximum = max;
    if (d_value > d_maximum)
        setValue(d_maximum);
}

void ScientificSpinBox::setRange(double min, double max)
{
    setMinimum(min);
    setMaximum(max);
}

double ScientificSpinBox::singleStep() const
{
    return d_singleStep;
}

void ScientificSpinBox::setSingleStep(double step)
{
    d_singleStep = std::abs(step);
}

ScientificSpinBox::StepMode ScientificSpinBox::stepMode() const
{
    return d_stepMode;
}

void ScientificSpinBox::setStepMode(StepMode mode)
{
    d_stepMode = mode;
}

double ScientificSpinBox::fixedStepSize() const
{
    return d_fixedStepSize;
}

void ScientificSpinBox::setFixedStepSize(double size)
{
    d_fixedStepSize = std::abs(size);
}

int ScientificSpinBox::displayPrecision() const
{
    return d_displayPrecision;
}

void ScientificSpinBox::setDisplayPrecision(int precision)
{
    d_displayPrecision = std::clamp(precision, -1, MAX_PRECISION);
    d_precisionExplicit = (d_displayPrecision >= 0);
    updateDisplayText();
    updateEditText();

    if (!d_isEditing)
        lineEdit()->setText(d_cachedDisplayText);
}

ScientificSpinBox::DisplayMode ScientificSpinBox::displayMode() const
{
    return d_displayMode;
}

void ScientificSpinBox::setDisplayMode(DisplayMode mode)
{
    if (d_displayMode == mode)
        return;
    d_displayMode = mode;
    updateDisplayText();
    if (!d_isEditing)
        lineEdit()->setText(d_cachedDisplayText);
}

QString ScientificSpinBox::suffix() const
{
    return d_suffix;
}

void ScientificSpinBox::setSuffix(const QString &suffix)
{
    d_suffix = suffix;
    lineEdit()->setMaxLength(MAX_INPUT_LENGTH + d_suffix.length());
    updateDisplayText();
    if (!d_isEditing)
        lineEdit()->setText(d_cachedDisplayText);
}

void ScientificSpinBox::stepBy(int steps)
{
    if (steps == 0)
        return;

    Qt::KeyboardModifiers modifiers = (d_currentWheelModifiers != Qt::NoModifier)
                                     ? d_currentWheelModifiers
                                     : QApplication::keyboardModifiers();

    if (d_stepMode == StepMode::Fixed && d_fixedStepSize > 0.0) {
        double base = d_fixedStepSize;
        if (modifiers & Qt::ControlModifier)
            base *= 10.0;
        setValue(d_value + steps * base);
        return;
    }

    if (modifiers & Qt::ControlModifier) {
        if (qFuzzyIsNull(d_value))
            setValue(steps > 0 ? 1.0 : -1.0);
        else
            setValue(d_value * std::pow(2.0, steps));
        return;
    }

    setValue(d_value + steps * calculateStepSize());
}

QValidator::State ScientificSpinBox::validate(QString &input, int &pos) const
{
    Q_UNUSED(pos)

    initializeValidator();

    QString cleanInput = stripSuffix(removeSuperscript(input));
    return d_validator->validate(cleanInput, pos);
}

void ScientificSpinBox::fixup(QString &input) const
{
    initializeValidator();

    QString cleanInput = stripSuffix(removeSuperscript(input));
    cleanInput.replace(QString("×"), QString("e"));
    cleanInput.replace(QString("*"), QString("e"));

    bool ok;
    double value = cleanInput.toDouble(&ok);
    if (ok)
        input = formatForEdit(value);
}

QSize ScientificSpinBox::sizeHint() const
{
    QSize size = QAbstractSpinBox::sizeHint();
    const QFontMetrics fm(fontMetrics());

    // Wide enough for fixed-point ("-123456.000000") and scientific ("-1.23456 × 10^(-45)") plus suffix
    const int fixedWidth = fm.horizontalAdvance(u"-123456.000000"_s + d_suffix);
    const int sciWidth = fm.horizontalAdvance(u"-1.23456 × 10^(-45)"_s + d_suffix);
    const int width = std::max(fixedWidth, sciWidth) + 50;

    size.setWidth(std::max(width, size.width()));
    return size;
}

QSize ScientificSpinBox::minimumSizeHint() const
{
    QSize size = QAbstractSpinBox::minimumSizeHint();
    const QFontMetrics fm(fontMetrics());

    const int width = fm.horizontalAdvance(u"-1 × 10^(-9)"_s + d_suffix) + 40;
    size.setWidth(std::max(width, size.width()));
    return size;
}

void ScientificSpinBox::selectAll()
{
    lineEdit()->selectAll();
}

void ScientificSpinBox::clear()
{
    setValue(0.0);
}

QLineEdit* ScientificSpinBox::lineEdit() const
{
    return QAbstractSpinBox::lineEdit();
}

QAbstractSpinBox::StepEnabled ScientificSpinBox::stepEnabled() const
{
    StepEnabled enabled = StepNone;

    if (d_value > d_minimum || qFuzzyCompare(d_value, d_minimum))
        enabled |= StepDownEnabled;

    if (d_value < d_maximum || qFuzzyCompare(d_value, d_maximum))
        enabled |= StepUpEnabled;

    return enabled;
}

void ScientificSpinBox::keyPressEvent(QKeyEvent *event)
{
    if (event->key() == Qt::Key_Up) {
        stepBy(1);
        return;
    } else if (event->key() == Qt::Key_Down) {
        stepBy(-1);
        return;
    }

    QAbstractSpinBox::keyPressEvent(event);
}

void ScientificSpinBox::focusInEvent(QFocusEvent *event)
{
    d_isEditing = true;
    updateEditText();
    lineEdit()->setText(d_cachedEditText);

    QAbstractSpinBox::focusInEvent(event);

    QMetaObject::invokeMethod(this, "selectAll", Qt::QueuedConnection);
}

void ScientificSpinBox::focusOutEvent(QFocusEvent *event)
{
    d_isEditing = false;
    lineEdit()->setStyleSheet({});

    QString currentText = lineEdit()->text();
    double newValue = valueFromText(currentText);

    if (!qFuzzyCompare(newValue, d_value)) {
        if (d_displayPrecision < 0) {
            int detectedPrecision = detectPrecision(currentText);
            d_displayPrecision = detectedPrecision;
        }
        setValue(newValue);
    } else {
        if (d_displayPrecision < 0) {
            int detectedPrecision = detectPrecision(currentText);
            if (detectedPrecision != detectPrecision(d_cachedDisplayText)) {
                d_displayPrecision = detectedPrecision;
                updateDisplayText();
                lineEdit()->setText(d_cachedDisplayText);
            }
        }

        if (d_displayPrecision >= 0) {
            updateDisplayText();
            lineEdit()->setText(d_cachedDisplayText);
        }
    }

    QAbstractSpinBox::focusOutEvent(event);
}

void ScientificSpinBox::wheelEvent(QWheelEvent *event)
{
    if (!hasFocus()) {
        event->ignore();
        return;
    }

    const int delta = event->angleDelta().y();
    const int steps = delta / 120;

    d_currentWheelModifiers = event->modifiers();
    stepBy(steps);
    d_currentWheelModifiers = Qt::NoModifier;

    bool wasEditing = d_isEditing;
    d_isEditing = false;
    updateDisplayText();
    lineEdit()->setText(d_cachedDisplayText);
    d_isEditing = wasEditing;

    event->accept();
}

void ScientificSpinBox::contextMenuEvent(QContextMenuEvent *event)
{
    QMenu *menu = lineEdit()->createStandardContextMenu();
    menu->addSeparator();

    QMenu *modeMenu = menu->addMenu(tr("Display Mode"));
    QActionGroup *modeGroup = new QActionGroup(modeMenu);
    modeGroup->setExclusive(true);

    auto addModeAction = [&](const QString &text, DisplayMode mode) {
        QAction *action = modeMenu->addAction(text);
        action->setCheckable(true);
        action->setChecked(d_displayMode == mode);
        modeGroup->addAction(action);
        connect(action, &QAction::triggered, this, [this, mode]() { setDisplayMode(mode); });
    };
    addModeAction(tr("Auto"), DisplayMode::Auto);
    addModeAction(tr("Fixed"), DisplayMode::Fixed);
    addModeAction(tr("Scientific"), DisplayMode::Scientific);

    menu->addSeparator();

    QMenu *stepMenu = menu->addMenu(tr("Step Size"));
    QActionGroup *stepGroup = new QActionGroup(stepMenu);
    stepGroup->setExclusive(true);

    QAction *adaptiveAction = stepMenu->addAction(tr("Adaptive"));
    adaptiveAction->setCheckable(true);
    adaptiveAction->setChecked(d_stepMode == StepMode::Adaptive);
    stepGroup->addAction(adaptiveAction);
    connect(adaptiveAction, &QAction::triggered, this, [this]() {
        setStepMode(StepMode::Adaptive);
    });

    QWidget *fixedStepWidget = new QWidget;
    QHBoxLayout *fixedStepLayout = new QHBoxLayout(fixedStepWidget);
    fixedStepLayout->setContentsMargins(20, 2, 8, 2);
    QAction *fixedAction = new QAction(tr("Fixed:"), fixedStepWidget);
    fixedAction->setCheckable(true);
    fixedAction->setChecked(d_stepMode == StepMode::Fixed);
    stepGroup->addAction(fixedAction);
    fixedStepLayout->addWidget(new QLabel(tr("Fixed:")));
    QLineEdit *fixedStepEdit = new QLineEdit;
    QDoubleValidator *fixedStepValidator = new QDoubleValidator(fixedStepEdit);
    fixedStepValidator->setBottom(std::numeric_limits<double>::min());
    fixedStepValidator->setTop(std::numeric_limits<double>::max());
    fixedStepValidator->setDecimals(15);
    fixedStepValidator->setNotation(QDoubleValidator::ScientificNotation);
    fixedStepEdit->setValidator(fixedStepValidator);

    {
        char buf[64];
        const double sz = d_fixedStepSize;
        QString initText;
        if (qFuzzyIsNull(sz)) {
            initText = u"0"_s;
        } else {
            auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf), sz, std::chars_format::scientific);
            initText = QString::fromLatin1(buf, static_cast<int>(ptr - buf));
        }
        fixedStepEdit->setText(initText);
    }

    fixedStepLayout->addWidget(fixedStepEdit);
    QWidgetAction *fixedStepWidgetAction = new QWidgetAction(stepMenu);
    fixedStepWidgetAction->setDefaultWidget(fixedStepWidget);
    stepMenu->addAction(fixedStepWidgetAction);

    connect(fixedAction, &QAction::triggered, this, [this]() {
        setStepMode(StepMode::Fixed);
    });

    connect(fixedStepEdit, &QLineEdit::editingFinished, this, [this, fixedStepEdit]() {
        bool ok = false;
        double v = fixedStepEdit->text().toDouble(&ok);
        if (ok && v > 0.0) {
            setFixedStepSize(v);
            setStepMode(StepMode::Fixed);
        }
    });
    connect(fixedStepEdit, &QLineEdit::returnPressed, this, [this, fixedStepEdit]() {
        bool ok = false;
        double v = fixedStepEdit->text().toDouble(&ok);
        if (ok && v > 0.0) {
            setFixedStepSize(v);
            setStepMode(StepMode::Fixed);
        }
    });

    menu->addSeparator();

    QWidget *precisionWidget = new QWidget;
    QHBoxLayout *precisionLayout = new QHBoxLayout(precisionWidget);
    precisionLayout->setContentsMargins(20, 2, 8, 2);
    precisionLayout->addWidget(new QLabel(tr("Decimal places:")));
    QSpinBox *precisionBox = new QSpinBox;
    precisionBox->setRange(-1, MAX_PRECISION);
    precisionBox->setSpecialValueText(tr("Auto"));
    precisionBox->setValue(d_displayPrecision);
    precisionLayout->addWidget(precisionBox);
    QWidgetAction *precisionAction = new QWidgetAction(menu);
    precisionAction->setDefaultWidget(precisionWidget);
    menu->addAction(precisionAction);

    connect(precisionBox, &QSpinBox::valueChanged, this, [this](int v) {
        setDisplayPrecision(v);
    });

    menu->addSeparator();

    QAction *copyAction = menu->addAction(tr("Copy Value"));
    connect(copyAction, &QAction::triggered, this, [this]() {
        QApplication::clipboard()->setText(QString::number(d_value, 'g', 15));
    });

    menu->exec(event->globalPos());
    delete menu;
}

QString ScientificSpinBox::textFromValue(double value) const
{
    if (d_isEditing)
        return formatForEdit(value);
    else
        return formatForDisplay(value);
}

double ScientificSpinBox::valueFromText(const QString &text) const
{
    QString cleanText = stripSuffix(removeSuperscript(text));
    cleanText.replace(QString("×"), QString("e"));
    cleanText.replace(QString("*"), QString("e"));

    bool ok;
    double value = cleanText.toDouble(&ok);

    if (!ok)
        return d_value;

    return std::clamp(value, d_minimum, d_maximum);
}

void ScientificSpinBox::onEditingFinished()
{
    QString currentText = lineEdit()->text();
    double newValue = valueFromText(currentText);

    if (d_displayPrecision < 0) {
        int detectedPrecision = detectPrecision(currentText);
        d_displayPrecision = detectedPrecision;
    }

    if (!qFuzzyCompare(newValue, d_value)) {
        setValue(newValue);
    } else {
        updateDisplayText();
        if (!d_isEditing)
            lineEdit()->setText(d_cachedDisplayText);
    }

    emit editingFinished();
}

void ScientificSpinBox::onTextChanged()
{
    if (!d_isEditing)
        return;

    if (isValidInput(lineEdit()->text()))
        lineEdit()->setStyleSheet({});
    else
        lineEdit()->setStyleSheet(u"background-color: %1;"_s.arg(
            ThemeColors::getCSSColor(ThemeColors::StatusError, this)));
}

void ScientificSpinBox::initializeValidator() const
{
    if (!d_validator) {
        d_validator = std::make_unique<QDoubleValidator>(d_minimum, d_maximum, 16);
        d_validator->setNotation(QDoubleValidator::ScientificNotation);
    } else {
        d_validator->setRange(d_minimum, d_maximum, 16);
    }
}

void ScientificSpinBox::updateDisplayText()
{
    d_cachedDisplayText = formatForDisplay(d_value);
}

void ScientificSpinBox::updateEditText()
{
    d_cachedEditText = formatForEdit(d_value);
}

QString ScientificSpinBox::formatForDisplay(double value) const
{
    using BC::Gui::NumericDisplayMode;
    NumericDisplayMode mode;
    switch (d_displayMode) {
    case DisplayMode::Fixed:      mode = NumericDisplayMode::Fixed;      break;
    case DisplayMode::Scientific: mode = NumericDisplayMode::Scientific;  break;
    default:                      mode = NumericDisplayMode::Auto;        break;
    }
    return BC::Gui::formatNumberForDisplay(value, d_displayPrecision, mode) + d_suffix;
}

QString ScientificSpinBox::formatForEdit(double value) const
{
    if (qFuzzyIsNull(value))
        return u"0"_s;

    int precision = d_displayPrecision;
    if (precision < 0)
        precision = 6;

    return QString::number(value, 'e', precision);
}

int ScientificSpinBox::detectPrecision(const QString &text) const
{
    QString cleanText = stripSuffix(removeSuperscript(text.trimmed()));

    int decimalPos = cleanText.indexOf('.');
    if (decimalPos == -1)
        return 0;

    int ePos = cleanText.indexOf('e', Qt::CaseInsensitive);
    if (ePos == -1) {
        QString afterDecimal = cleanText.mid(decimalPos + 1);
        while (afterDecimal.endsWith('0') && afterDecimal.length() > 1)
            afterDecimal.chop(1);
        return afterDecimal.length();
    }

    QString mantissaAfterDecimal = cleanText.mid(decimalPos + 1, ePos - decimalPos - 1);
    while (mantissaAfterDecimal.endsWith('0') && mantissaAfterDecimal.length() > 1)
        mantissaAfterDecimal.chop(1);
    return mantissaAfterDecimal.length();
}

double ScientificSpinBox::calculateStepSize() const
{
    if (qFuzzyIsNull(d_value))
        return 1.0;

    const double absValue = std::abs(d_value);
    const bool showFixed = (d_displayMode == DisplayMode::Fixed) ||
                           (d_displayMode == DisplayMode::Auto && absValue >= 1e-6 && absValue < 1e6);

    const int effectivePrecision = (d_displayPrecision >= 0)
        ? d_displayPrecision
        : (showFixed ? detectFixedPrecision(absValue) : detectSciPrecision(absValue));

    if (showFixed) {
        // Step = place value of last digit: 10^(-decimal_places)
        return std::pow(10.0, -effectivePrecision);
    } else {
        const int exponent = static_cast<int>(std::floor(std::log10(absValue)));
        // Step = place value of last mantissa digit: 10^(exponent - precision)
        return std::pow(10.0, exponent - effectivePrecision);
    }
}

QString ScientificSpinBox::applySuperscript(const QString &text) const
{
    return BC::Gui::formatScientificSuperscript(text);
}

QString ScientificSpinBox::removeSuperscript(const QString &text) const
{
    QString result = text;

    QRegularExpression pattern("\\s*×\\s*10\\^\\(([+-]?\\d+)\\)");
    QRegularExpressionMatch match = pattern.match(result);

    if (match.hasMatch()) {
        result.replace(match.captured(0), QString("e%1").arg(match.captured(1)));
    } else {
        QRegularExpression simplePattern("\\s*×\\s*10\\^([+-]?\\d+)");
        QRegularExpressionMatch simpleMatch = simplePattern.match(result);

        if (simpleMatch.hasMatch())
            result.replace(simpleMatch.captured(0), QString("e%1").arg(simpleMatch.captured(1)));
    }

    // Fallback: handle old Unicode superscript format if present
    result.replace(QString("⁰"), QString("0"));
    result.replace(QString("¹"), QString("1"));
    result.replace(QString("²"), QString("2"));
    result.replace(QString("³"), QString("3"));
    result.replace(QString("⁴"), QString("4"));
    result.replace(QString("⁵"), QString("5"));
    result.replace(QString("⁶"), QString("6"));
    result.replace(QString("⁷"), QString("7"));
    result.replace(QString("⁸"), QString("8"));
    result.replace(QString("⁹"), QString("9"));
    result.replace(QString("⁺"), QString("+"));
    result.replace(QString("⁻"), QString("-"));

    result.replace(QString(" × 10"), QString("e"));
    result.replace(QString(" x 10"), QString("e"));
    result.replace(QString("×10"), QString("e"));
    result.replace(QString("x10"), QString("e"));

    return result;
}

QString ScientificSpinBox::stripSuffix(const QString &text) const
{
    if (d_suffix.isEmpty())
        return text;
    QString result = text.trimmed();
    if (result.endsWith(d_suffix))
        result.chop(d_suffix.length());
    return result;
}

int ScientificSpinBox::detectSciPrecision(double value) const
{
    char buf[64];
    auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf), value, std::chars_format::scientific);
    const char *dot = static_cast<const char *>(std::memchr(buf, '.', ptr - buf));
    if (!dot)
        return 0;
    const char *e = static_cast<const char *>(std::memchr(dot, 'e', ptr - dot));
    if (!e)
        return 0;
    return static_cast<int>(e - dot - 1);
}

int ScientificSpinBox::detectFixedPrecision(double value) const
{
    char buf[64];
    auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf), value, std::chars_format::fixed);
    const char *dot = static_cast<const char *>(std::memchr(buf, '.', ptr - buf));
    if (!dot)
        return 0;
    return static_cast<int>(ptr - dot - 1);
}

bool ScientificSpinBox::isValidInput(const QString &text) const
{
    int pos = 0;
    QString testText = text;
    return validate(testText, pos) != QValidator::Invalid;
}

void ScientificSpinBox::emitValueChanged()
{
    emit valueChanged(d_value);
}
