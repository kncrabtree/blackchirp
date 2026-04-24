#ifndef RUNTIMEHARDWARECONFIGDIALOG_H
#define RUNTIMEHARDWARECONFIGDIALOG_H

#include <QDialog>
#include <QTreeWidgetItem>
#include <QStringList>
#include <QPair>
#include <QMap>
#include <optional>
#include <hardware/core/runtimehardwareconfig.h>
#include <data/bcglobals.h>

// Forward declarations
class QListWidgetItem;
class LibraryStatusWidget;
class PythonSettingsWidget;

namespace Ui {
class RuntimeHardwareConfigDialog;
}

/**
 * @brief Runtime Hardware Configuration Dialog
 *
 * Provides user interface for selecting and configuring hardware implementations
 * at runtime. Uses a three-panel layout (overview, browser, configuration) for
 * hardware profile management, plus a Library Status tab for vendor libraries.
 */
class RuntimeHardwareConfigDialog : public QDialog
{
    Q_OBJECT

public:
    explicit RuntimeHardwareConfigDialog(QWidget *parent = nullptr);
    ~RuntimeHardwareConfigDialog();

    bool openFtmwConfigOnClose() const { return d_openFtmwConfigOnClose; }

private slots:
    void onHardwareBrowserSelectionChanged(QListWidgetItem* current, QListWidgetItem* previous);

private:
    void populateConfigurationOverview();
    void refreshConfigurationOverview();
    void populateHardwareBrowser();
    void updateSelectionDisplay(const QString& selectedHardwareType);
    void updateRightPanelForHardwareType(const QString& hardwareType);

    // Profile selection state management
    void onProfileSelectionChanged(const QString& hardwareType);
    void onEnableToggled(const QString& hardwareType, bool enabled);
    void onProfileRadioClicked(const QString& hardwareType);
    void onProfileCheckboxClicked(const QString& hardwareType);
    void updatePreviewConfiguration();

    // Profile add/remove
    void onAddProfile(const QString& hardwareType);
    void onRemoveProfile(const QString& hardwareType);

    // Dialog accept/reject
    void onDialogAccepted();
    void onDialogRejected();

    // Validation
    void validatePreviewConfiguration();
    void updateValidationStatus(const QString& message, const QString& state);

    // Library status tab indicator
    void onLibraryStagingStateChanged(bool hasChanges);

    // Loadout management
    void populateLoadoutCombo();
    void switchToLoadout(const QString &name);
    void updateLoadoutButtonStates();
    void ensureRequiredTypes();
    void onLoadoutComboChanged(const QString &name);
    void onLoadoutSave();
    void onLoadoutSaveAs();
    void onLoadoutDelete();
    void onLoadoutSetDefault();

    struct ProfileOverrides {
        std::optional<bool> threaded;
        std::optional<QString> pythonScript;
        std::optional<QString> pythonClass;
        std::optional<QString> pythonEnv;
    };

    Ui::RuntimeHardwareConfigDialog *pu_ui;

    // Preview state
    std::map<QString, QString, std::less<>> d_originalRuntimeConfig;
    std::map<QString, QString, std::less<>> d_previewRuntimeConfig;
    std::map<QString, ProfileOverrides, std::less<>> d_profileOverrides;
    QString d_currentHardwareType;

    // Loadout state
    QString d_activeLoadoutName;
    bool d_openFtmwConfigOnClose{false};

    static bool getTypeDefaultThreaded(const QString& hardwareType);

    // Library status widget (owned by tab layout)
    LibraryStatusWidget *p_libraryStatusWidget;
};

#endif // RUNTIMEHARDWARECONFIGDIALOG_H
