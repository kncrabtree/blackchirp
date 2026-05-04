#ifndef SCIENTIFICINPUTWIDGET_H
#define SCIENTIFICINPUTWIDGET_H

#include <QWidget>
#include <QVBoxLayout>
#include <QLabel>
#include "scientificspinbox.h"

/**
 * @brief A wrapper widget for ScientificSpinBox with validation and warning display
 * 
 * ScientificInputWidget provides:
 * - A ScientificSpinBox for scientific notation input
 * - A warning label that appears when input is invalid
 * - Signal blocking when input is invalid (returns last valid value)
 * - Clear visual feedback for validation state
 * 
 * When input is invalid:
 * - Warning label shows with last valid value
 * - valueChanged signals are blocked
 * - value() returns the last valid value
 * - Warning is hidden when input becomes valid again
 */
class ScientificInputWidget : public QWidget
{
    Q_OBJECT
    Q_PROPERTY(double value READ value WRITE setValue NOTIFY valueChanged USER true)
    Q_PROPERTY(double minimum READ minimum WRITE setMinimum)
    Q_PROPERTY(double maximum READ maximum WRITE setMaximum)
    Q_PROPERTY(double singleStep READ singleStep WRITE setSingleStep)

public:
    explicit ScientificInputWidget(QWidget *parent = nullptr);
    ~ScientificInputWidget() override = default;

    // Value access (returns last valid value if current input is invalid)
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
    
    // Validation state
    bool isValid() const;
    QString lastValidValueText() const;
    
    // Access to underlying spinbox (for advanced configuration)
    ScientificSpinBox* spinBox() const;
    
    // Keyboard tracking control
    void setKeyboardTracking(bool enabled);
    bool keyboardTracking() const;
    
    // Focus control
    void setFocus();
    bool hasFocus() const;

public slots:
    void selectAll();
    void clear();

signals:
    void valueChanged(double value);
    void editingFinished();
    void validationStateChanged(bool isValid);

protected:
    void resizeEvent(QResizeEvent *event) override;

private slots:
    void onSpinBoxValueChanged(double value);
    void onSpinBoxEditingFinished();
    void onSpinBoxTextChanged();
    void validateCurrentInput();

private:
    // UI components
    ScientificSpinBox *p_spinBox;
    QLabel *p_warningLabel;
    QVBoxLayout *p_layout;
    
    // Validation state
    bool d_isValid{true};
    double d_lastValidValue{0.0};
    QString d_lastValidText{"0"};
    bool d_keyboardTracking{true};
    bool d_signalsBlocked{false};
    
    // Helper methods
    void setupUI();
    void setupConnections();
    void updateValidationState(bool isValid);
    void showWarning(const QString &message);
    void hideWarning();
    void blockValueSignals(bool block);
    QString formatWarningMessage(double lastValidValue) const;
    
    // Style management
    void updateWarningStyle();
};

#endif // SCIENTIFICINPUTWIDGET_H