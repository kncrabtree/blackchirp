#ifndef OVERLAYBASEOPTIONSWIDGET_H
#define OVERLAYBASEOPTIONSWIDGET_H

#include <QWidget>
#include <QLineEdit>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QStringList>
#include <memory>

#include <data/experiment/overlaybase.h>

class OverlayBaseOptionsWidget : public QWidget
{
    Q_OBJECT

public:
    explicit OverlayBaseOptionsWidget(const QStringList &plotNames, QWidget *parent = nullptr);

    // Getters
    QString getLabel() const;
    QString getPlotId() const;
    double getYScale() const;
    double getYOffset() const;
    double getXOffset() const;

    // Setters
    void setLabel(const QString &label);
    void setPlotId(const QString &plotId);
    void setYScale(double yScale);
    void setYOffset(double yOffset);
    void setXOffset(double xOffset);

    // Validation
    bool validateSettings(QString &errorMessage, const QVector<std::shared_ptr<OverlayBase>> &existingOverlays) const;

    // Apply settings to overlay
    void applyToOverlay(std::shared_ptr<OverlayBase> overlay) const;

private:
    // UI elements
    QLineEdit *p_labelLineEdit;
    QComboBox *p_plotIdComboBox;
    QDoubleSpinBox *p_yScaleSpinBox;
    QDoubleSpinBox *p_yOffsetSpinBox;
    QDoubleSpinBox *p_xOffsetSpinBox;

    void setupUI();
    void initializeDefaults();
};

#endif // OVERLAYBASEOPTIONSWIDGET_H