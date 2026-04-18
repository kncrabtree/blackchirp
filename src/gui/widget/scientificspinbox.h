#ifndef SCIENTIFICSPINBOX_H
#define SCIENTIFICSPINBOX_H

#include <QAbstractSpinBox>
#include <QDoubleValidator>
#include <QKeyEvent>
#include <memory>
#include <limits>

class QContextMenuEvent;

/**
 * @brief A spinbox for scientific notation input and display
 *
 * Accepts input in both fixed and scientific notation (max 16 digits + decimal + sign = 18 chars).
 * Display modes: Auto (fixed for |value| in [1e-6, 1e6), scientific otherwise), Fixed, Scientific.
 * Switches to standard edit notation when focused for editing.
 * Precision-aware stepping; Ctrl modifier doubles or halves the value (×2 per step).
 * Optional unit suffix displayed alongside the value.
 */
class ScientificSpinBox : public QAbstractSpinBox
{
    Q_OBJECT
    Q_PROPERTY(double value READ value WRITE setValue NOTIFY valueChanged USER true)
    Q_PROPERTY(double minimum READ minimum WRITE setMinimum)
    Q_PROPERTY(double maximum READ maximum WRITE setMaximum)
    Q_PROPERTY(double singleStep READ singleStep WRITE setSingleStep)

public:
    enum class DisplayMode { Auto, Fixed, Scientific };
    enum class StepMode { Adaptive, Fixed };

    explicit ScientificSpinBox(QWidget *parent = nullptr);
    ~ScientificSpinBox() override = default;

    // Value access
    double value() const;
    void setValue(double value);

    // Range control
    double minimum() const;
    void setMinimum(double min);
    double maximum() const;
    void setMaximum(double max);
    void setRange(double min, double max);

    // Step control
    double singleStep() const;
    void setSingleStep(double step);

    // Step mode control
    StepMode stepMode() const;
    void setStepMode(StepMode mode);
    double fixedStepSize() const;
    void setFixedStepSize(double size);

    // Precision control
    int displayPrecision() const;
    void setDisplayPrecision(int precision);

    // Display mode control
    DisplayMode displayMode() const;
    void setDisplayMode(DisplayMode mode);

    // Suffix / unit label
    QString suffix() const;
    void setSuffix(const QString &suffix);

    // Override QAbstractSpinBox interface
    void stepBy(int steps) override;
    QValidator::State validate(QString &input, int &pos) const override;
    void fixup(QString &input) const override;

    // Size hint
    QSize sizeHint() const override;
    QSize minimumSizeHint() const override;

    // Public text conversion methods (needed by ScientificInputWidget)
    QString textFromValue(double value) const;
    double valueFromText(const QString &text) const;

    // Access to line edit for external validation
    QLineEdit* lineEdit() const;

public slots:
    void selectAll();
    void clear() override;

signals:
    void valueChanged(double value);
    void editingFinished();

protected:
    // Override QAbstractSpinBox virtual methods
    QAbstractSpinBox::StepEnabled stepEnabled() const override;

    // Override QWidget event handlers
    void keyPressEvent(QKeyEvent *event) override;
    void focusInEvent(QFocusEvent *event) override;
    void focusOutEvent(QFocusEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;
    void contextMenuEvent(QContextMenuEvent *event) override;

private slots:
    void onEditingFinished();
    void onTextChanged();

private:
    // Core value and validation
    double d_value{0.0};
    double d_minimum{-std::numeric_limits<double>::max()};
    double d_maximum{std::numeric_limits<double>::max()};
    double d_singleStep{1.0};
    int d_displayPrecision{-1}; // -1 = automatic detection
    bool d_precisionExplicit{false};
    DisplayMode d_displayMode{DisplayMode::Auto};
    StepMode d_stepMode{StepMode::Adaptive};
    double d_fixedStepSize{0.0};
    QString d_suffix;

    // State tracking
    bool d_isEditing{false};
    QString d_cachedDisplayText;
    QString d_cachedEditText;
    Qt::KeyboardModifiers d_currentWheelModifiers{Qt::NoModifier};
    mutable std::unique_ptr<QDoubleValidator> d_validator;

    // Helper methods
    void initializeValidator() const;
    void updateDisplayText();
    void updateEditText();
    QString formatForDisplay(double value) const;
    QString formatForEdit(double value) const;
    int detectPrecision(const QString &text) const;
    double calculateStepSize() const;
    QString applySuperscript(const QString &text) const;
    QString removeSuperscript(const QString &text) const;
    QString stripSuffix(const QString &text) const;
    bool isValidInput(const QString &text) const;
    void emitValueChanged();
    int detectSciPrecision(double value) const;
    int detectFixedPrecision(double value) const;

    // Constants
    static constexpr int MAX_INPUT_LENGTH = 18; // 16 digits + decimal + sign
    static constexpr int MAX_PRECISION = 15;
};

#endif // SCIENTIFICSPINBOX_H
