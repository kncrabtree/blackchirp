#ifndef OVERLAYCONFIGDIALOG_H
#define OVERLAYCONFIGDIALOG_H

#include <QDialog>
#include <QDialogButtonBox>
#include <QLabel>
#include <QVBoxLayout>
#include <QGroupBox>
#include <QPushButton>
#include <memory>

#include <data/experiment/overlaytypes.h>
#include "overlaybaseoptionswidget.h"

class FtmwViewWidget;

class OverlayConfigDialog : public QDialog
{
    Q_OBJECT

public:
    explicit OverlayConfigDialog(FtmwViewWidget *parent = nullptr);
    virtual ~OverlayConfigDialog() = default;

    // Setup UI after construction is complete - must be called by creator
    void setupUI();
    

    // Pure virtual method for overlay creation - must be implemented by derived classes
    virtual std::shared_ptr<OverlayBase> createOverlay() const = 0;

public slots:
    void accept() override;

protected:
    // Access to common UI elements
    OverlayBaseOptionsWidget *p_overlayOptionsWidget;
    QDialogButtonBox *p_buttonBox;
    QLabel *p_validationLabel;
    
    // Common parameters (accessed from parent widget)
    QStringList d_plotNames;
    double d_xRangeMin, d_xRangeMax;
    QVector<std::shared_ptr<OverlayBase>> d_existingOverlays;
    
    // Protected methods for derived classes to use
    void updateValidationStatus(bool valid, const QString &message = QString());
    void updateOkButtonState();
    
    // Virtual methods for derived classes to implement
    virtual void setupTypeSpecificUI() = 0;
    virtual void setupTypeSpecificConnections() = 0;
    virtual void initializeTypeSpecificDefaults() = 0;
    virtual bool validateTypeSpecificSettings(QString &errorMessage) = 0;
    virtual bool isTypeSpecificDataValid() const = 0;

private:
    void setupOverlayBaseOptions();
    void setupCommonConnections();
    void initializeCommonDefaults();

private slots:
    void onDialogAccepted();
};

#endif // OVERLAYCONFIGDIALOG_H