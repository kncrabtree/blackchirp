#ifndef SCIENTIFICSPINBOX_H
#define SCIENTIFICSPINBOX_H

#include <QAbstractSpinBox>
#include <QDoubleValidator>
#include <QKeyEvent>
#include <memory>
#include <limits>

/**
 * @brief A spinbox for scientific notation input and display
 * 
 * ScientificSpinBox is a specialized spinbox that:
 * - Accepts input in both fixed and scientific notation (max 16 digits + decimal + sign = 18 chars)
 * - Displays values in scientific notation with Unicode formatting (1.00 × 10^-3)
 * - Switches to standard notation (1.00e-3) when focused for editing
 * - Provides precision-aware stepping based on the last decimal digit
 * - Supports Ctrl modifier to multiply step size by 10
 * 
 * Examples of display behavior:
 * - 100 → 1e2
 * - 100. → 1.00e2  
 * - 100.0 → 1.000e2
 * - 0.0001 → 1e-4
 * - 0.000100 → 1.00e-4
 */
class ScientificSpinBox : public QAbstractSpinBox
{
    Q_OBJECT
    Q_PROPERTY(double value READ value WRITE setValue NOTIFY valueChanged USER true)
    Q_PROPERTY(double minimum READ minimum WRITE setMinimum)
    Q_PROPERTY(double maximum READ maximum WRITE setMaximum)
    Q_PROPERTY(double singleStep READ singleStep WRITE setSingleStep)

public:
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
    
    // Precision control
    int displayPrecision() const;
    void setDisplayPrecision(int precision);
    
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
    void clear();

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
    
    // Text conversion methods (now public above)

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
    bool isValidInput(const QString &text) const;
    void emitValueChanged();
    
    // Constants
    static constexpr int MAX_INPUT_LENGTH = 18; // 16 digits + decimal + sign
    static constexpr double STEP_MULTIPLIER = 10.0; // For Ctrl modifier
};

#endif // SCIENTIFICSPINBOX_H