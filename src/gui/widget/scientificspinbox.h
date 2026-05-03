#ifndef SCIENTIFICSPINBOX_H
#define SCIENTIFICSPINBOX_H

#include <QAbstractSpinBox>
#include <QDoubleValidator>
#include <QKeyEvent>
#include <memory>
#include <limits>

class QContextMenuEvent;

/// \brief Spin box accepting and displaying floating-point values in fixed or scientific notation.
class ScientificSpinBox : public QAbstractSpinBox
{
    Q_OBJECT
    Q_PROPERTY(double value READ value WRITE setValue NOTIFY valueChanged USER true)
    Q_PROPERTY(double minimum READ minimum WRITE setMinimum)
    Q_PROPERTY(double maximum READ maximum WRITE setMaximum)
    Q_PROPERTY(double singleStep READ singleStep WRITE setSingleStep)

public:
    /// \brief Controls how the value is formatted in the display (non-editing) state.
    enum class DisplayMode {
        Auto,       ///< Fixed notation when |value| is in [1e-6, 1e6); scientific otherwise.
        Fixed,      ///< Always fixed-point notation.
        Scientific  ///< Always scientific notation.
    };

    /// \brief Controls how the step size is calculated when the user increments or decrements.
    enum class StepMode {
        Adaptive, ///< Step size tracks the least-significant digit of the displayed value.
        Fixed     ///< Step size is the constant set by setFixedStepSize().
    };

    /// \brief Construct a ScientificSpinBox with the given parent widget.
    explicit ScientificSpinBox(QWidget *parent = nullptr);
    ~ScientificSpinBox() override = default;

    /// \brief Return the current value.
    double value() const;
    /// \brief Set the value, clamping to [minimum(), maximum()].
    void setValue(double value);

    /// \brief Return the lower bound of the allowed range.
    double minimum() const;
    /// \brief Set the lower bound; adjusts the current value if it falls below the new minimum.
    void setMinimum(double min);
    /// \brief Return the upper bound of the allowed range.
    double maximum() const;
    /// \brief Set the upper bound; adjusts the current value if it exceeds the new maximum.
    void setMaximum(double max);
    /// \brief Convenience; equivalent to calling setMinimum() then setMaximum().
    void setRange(double min, double max);

    /// \brief Return the nominal single-step size used in Adaptive mode.
    double singleStep() const;
    /// \brief Set the nominal single-step size used in Adaptive mode.
    void setSingleStep(double step);

    /// \brief Return the current step mode.
    StepMode stepMode() const;
    /// \brief Set the step mode.
    void setStepMode(StepMode mode);
    /// \brief Return the fixed step size used when stepMode() is StepMode::Fixed.
    double fixedStepSize() const;
    /// \brief Set the fixed step size; has no effect unless stepMode() is StepMode::Fixed.
    void setFixedStepSize(double size);

    /// \brief Return the display precision (number of decimal places), or -1 for automatic.
    int displayPrecision() const;
    /// \brief Set the display precision.
    ///
    /// Pass -1 to restore automatic precision detection. Valid range is [-1, 15].
    void setDisplayPrecision(int precision);

    /// \brief Return the current display mode.
    DisplayMode displayMode() const;
    /// \brief Set the display mode and refresh the displayed text.
    void setDisplayMode(DisplayMode mode);

    /// \brief Return the unit suffix appended to the displayed text.
    QString suffix() const;
    /// \brief Set the unit suffix (e.g. \c " MHz"). Pass an empty string to clear.
    void setSuffix(const QString &suffix);

    /// \brief Increment or decrement the value by \a steps logical steps.
    ///
    /// In Adaptive mode the step size tracks the least-significant digit of the
    /// displayed value. In Fixed mode the step size is fixedStepSize(). When the
    /// Ctrl modifier is held, Adaptive mode doubles or halves the value instead of
    /// adding a step, and Fixed mode multiplies the fixed step by 10.
    void stepBy(int steps) override;
    /// \brief Validate the line-edit contents; delegates to a QDoubleValidator.
    QValidator::State validate(QString &input, int &pos) const override;
    /// \brief Attempt to interpret \a input as a valid double and reformat it.
    void fixup(QString &input) const override;

    /// \brief Return the preferred size, wide enough to show typical fixed and scientific text.
    QSize sizeHint() const override;
    /// \brief Return the minimum acceptable size.
    QSize minimumSizeHint() const override;

    /// \brief Format \a value as display or edit text depending on the current editing state.
    QString textFromValue(double value) const;
    /// \brief Parse \a text (stripping superscript notation and the suffix) and return its double value.
    double valueFromText(const QString &text) const;

    /// \brief Return the underlying QLineEdit for external validation or styling.
    QLineEdit* lineEdit() const;

public slots:
    /// \brief Select all text in the line edit.
    void selectAll();
    /// \brief Reset the value to 0.0.
    void clear() override;

signals:
    /// \brief Emitted whenever the value changes.
    void valueChanged(double value);
    /// \brief Emitted when the user finishes editing (equivalent to QLineEdit::editingFinished).
    void editingFinished();

protected:
    /// \brief Return which step directions are enabled based on the current value and range.
    QAbstractSpinBox::StepEnabled stepEnabled() const override;

    /// \brief Handle Up/Down arrow keys for stepping; delegate everything else to QAbstractSpinBox.
    void keyPressEvent(QKeyEvent *event) override;
    /// \brief Switch to edit notation and select all text when focus is received.
    void focusInEvent(QFocusEvent *event) override;
    /// \brief Commit the edited value and switch back to display notation when focus is lost.
    void focusOutEvent(QFocusEvent *event) override;
    /// \brief Step the value on wheel events when the widget has focus; ignore otherwise.
    void wheelEvent(QWheelEvent *event) override;
    /// \brief Show a context menu offering display-mode, step-size, and precision controls.
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
