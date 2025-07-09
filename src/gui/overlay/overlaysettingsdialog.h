#ifndef OVERLAYSETTINGSDIALOG_H
#define OVERLAYSETTINGSDIALOG_H

#include <QDialog>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QDialogButtonBox>
#include <QLabel>
#include <memory>

#include <data/experiment/overlaybase.h>
#include <data/storage/settingsstorage.h>
#include <data/storage/overlaystorage.h>
#include "overlaybaseoptionswidget.h"

namespace BC::Key::OverlaySettings {
static const QString key{"OverlaySettingsDialog"};
static const QString geometry{"geometry"};
}

class OverlaySettingsDialog : public QDialog, public SettingsStorage
{
    Q_OBJECT

public:
    explicit OverlaySettingsDialog(std::shared_ptr<OverlayBase> overlay, 
                                  const QStringList &plotNames,
                                  double xRangeMin, double xRangeMax,
                                  std::shared_ptr<OverlayStorage> overlayStorage,
                                  QWidget *parent = nullptr);
    virtual ~OverlaySettingsDialog();

    // Setup UI after construction is complete - must be called by creator
    void setupUI();

public slots:
    void accept() override;
    void reject() override;

signals:
    void overlaySettingsChanged(std::shared_ptr<OverlayBase> overlay);

protected:
    // Virtual functions for type-specific extensions
    virtual void setupTypeSpecificUI() {}
    virtual void setupTypeSpecificConnections() {}
    virtual void loadTypeSpecificSettings() {}
    virtual void saveTypeSpecificSettings() {}
    virtual void resetTypeSpecificDefaults() {}
    virtual bool validateTypeSpecificSettings(QString &errorMessage) { Q_UNUSED(errorMessage) return true; }
    
    // Qt overrides
    void closeEvent(QCloseEvent *event) override;

    // Access to common components for derived classes
    std::shared_ptr<OverlayBase> d_overlay;
    QStringList d_plotNames;
    double d_xRangeMin;
    double d_xRangeMax;
    std::shared_ptr<OverlayStorage> p_overlayStorage;
    QVBoxLayout *p_mainLayout;
    OverlayBaseOptionsWidget *p_optionsWidget;

private slots:
    void onSettingsChanged();
    void onResetToDefaults();
    void onRealTimeSettingsChanged();

private:
    void setupConnections();
    void loadCurrentSettings();
    void saveCurrentSettings();

    QDialogButtonBox *p_buttonBox;
    QPushButton *p_resetButton;
    QLabel *p_titleLabel;
    
    // Store original settings for reset functionality
    QString d_originalLabel;
    QString d_originalPlotId;
    double d_originalYScale;
    double d_originalYOffset;
    double d_originalXOffset;
    bool d_originalMinFreqEnabled;
    double d_originalMinFreqValue;
    bool d_originalMaxFreqEnabled;
    double d_originalMaxFreqValue;
};

#endif // OVERLAYSETTINGSDIALOG_H