#ifndef OVERLAYBASEOPTIONSWIDGET_H
#define OVERLAYBASEOPTIONSWIDGET_H

#include <QWidget>
#include <QLineEdit>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QFormLayout>
#include <QLabel>
#include <QStringList>
#include <memory>

#include <data/experiment/overlaybase.h>

class OverlayBaseOptionsWidget : public QWidget
{
    Q_OBJECT

public:
    explicit OverlayBaseOptionsWidget(const QStringList &plotNames, 
                                     double xRangeMin = 0.0, double xRangeMax = 1000.0,
                                     QWidget *parent = nullptr);

    // Getters
    QString getLabel() const;
    QString getPlotId() const;
    double getYScale() const;
    double getYOffset() const;
    double getXOffset() const;
    bool getMinFreqEnabled() const;
    double getMinFreqValue() const;
    bool getMaxFreqEnabled() const;
    double getMaxFreqValue() const;

    // Setters
    void setLabel(const QString &label);
    void setPlotId(const QString &plotId);
    void setYScale(double yScale);
    void setYOffset(double yOffset);
    void setXOffset(double xOffset);
    void setMinFreqLimit(bool enabled, double value);
    void setMaxFreqLimit(bool enabled, double value);

    // Validation
    bool validateSettings(QString &errorMessage, const QVector<std::shared_ptr<OverlayBase>> &existingOverlays) const;

    // Apply settings to overlay
    void applyToOverlay(std::shared_ptr<OverlayBase> overlay) const;

private slots:
    void onLabelChanged();

private:
    // UI elements
    QLineEdit *p_labelLineEdit;
    QLabel *p_sanitizedLabelDisplay;
    QComboBox *p_plotIdComboBox;
    QDoubleSpinBox *p_yScaleSpinBox;
    QDoubleSpinBox *p_yOffsetSpinBox;
    QDoubleSpinBox *p_xOffsetSpinBox;
    QCheckBox *p_minFreqCheckBox;
    QDoubleSpinBox *p_minFreqSpinBox;
    QCheckBox *p_maxFreqCheckBox;
    QDoubleSpinBox *p_maxFreqSpinBox;
    
    // X range values for default initialization
    double d_xRangeMin, d_xRangeMax;

    void setupUI();
    void initializeDefaults();
    QString sanitizeLabel(const QString& label) const;
};

#endif // OVERLAYBASEOPTIONSWIDGET_H