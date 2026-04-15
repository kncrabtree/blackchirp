#ifndef OVERLAYBASEOPTIONSWIDGET_H
#define OVERLAYBASEOPTIONSWIDGET_H

#include <QWidget>
#include <QLineEdit>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QFormLayout>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QLabel>
#include <QStringList>
#include <QPushButton>
#include <QRegularExpressionValidator>
#include <memory>

#include <data/experiment/overlaybase.h>
#include <data/analysis/ft.h>
#include <data/storage/settingsstorage.h>
#include <gui/widget/scientificinputwidget.h>

// Namespace for settings keys
namespace BC::Key::OverlayBaseOptions {
inline constexpr QLatin1StringView key{"OverlayBaseOptionsWidget"};
inline constexpr QLatin1StringView autoscalePercentage{"autoscalePercentage"};
}

class OverlayBaseOptionsWidget : public QWidget, public SettingsStorage
{
    Q_OBJECT

public:
    explicit OverlayBaseOptionsWidget(const QStringList &plotNames, 
                                     const Ft &currentFt = Ft(),
                                     QWidget *parent = nullptr);

    // Getters
    QString getLabel() const;
    QString getPlotId() const;
    QString getComment() const;
    double getYScale() const;
    double getYOffset() const;
    double getXOffset() const;
    double getAutoscalePercentage() const;
    bool getMinFreqEnabled() const;
    double getMinFreqValue() const;
    bool getMaxFreqEnabled() const;
    double getMaxFreqValue() const;

    // Setters
    void setLabel(const QString &label);
    void setPlotId(const QString &plotId);
    void setComment(const QString &comment);
    void setYScale(double yScale);
    void setYOffset(double yOffset);
    void setXOffset(double xOffset);
    void setAutoscalePercentage(double percentage);
    void setMinFreqLimit(bool enabled, double value);
    void setMaxFreqLimit(bool enabled, double value);

    // Validation
    bool validateSettings(QString &errorMessage, const QVector<std::shared_ptr<OverlayBase>> &existingOverlays) const;

    // Apply settings to overlay
    void applyToOverlay(std::shared_ptr<OverlayBase> overlay) const;
    
    // Set overlay reference for autoscale functionality
    void setOverlayReference(std::shared_ptr<OverlayBase> overlay);

signals:
    void settingsChanged(); // Emitted when any setting changes (except label)
    void labelChanged(); // Emitted when label changes

private slots:
    void onLabelChanged();
    void onAutoscaleClicked();
    void onInvertClicked();

private:
    // UI elements
    QLineEdit *p_labelLineEdit;
    QLabel *p_sanitizedLabelDisplay;
    QLineEdit *p_commentLineEdit;
    QComboBox *p_plotIdComboBox;
    ScientificInputWidget *p_yScaleInputWidget;
    QPushButton *p_invertButton;
    QDoubleSpinBox *p_yOffsetSpinBox;
    QDoubleSpinBox *p_xOffsetSpinBox;
    QDoubleSpinBox *p_autoscalePercentageSpinBox;
    QPushButton *p_autoscaleButton;
    QCheckBox *p_minFreqCheckBox;
    QDoubleSpinBox *p_minFreqSpinBox;
    QCheckBox *p_maxFreqCheckBox;
    QDoubleSpinBox *p_maxFreqSpinBox;
    
    // References for autoscale functionality
    std::shared_ptr<OverlayBase> d_overlayRef;
    Ft d_currentFt;
    bool d_hasFtData{false};
    
    // X range values for default initialization
    double d_xRangeMin, d_xRangeMax;
    
    // Semicolon validator for comment field
    QRegularExpressionValidator *p_commentValidator;

    void setupUI();
    void initializeDefaults();
    QString sanitizeLabel(const QString& label) const;
    
    // Default values
    static constexpr double DEFAULT_AUTOSCALE_PERCENTAGE = 100.0;
};

#endif // OVERLAYBASEOPTIONSWIDGET_H