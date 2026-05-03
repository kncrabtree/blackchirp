#ifndef HWSETTINGSWIDGET_H
#define HWSETTINGSWIDGET_H

#include <QWidget>
#include <QHash>
#include <QMap>
#include <data/storage/settingsstorage.h>
#include <hardware/core/hardwareregistry.h>

class QFormLayout;
class QGroupBox;
class QLabel;
class QTableWidget;
class QTabWidget;

/*!
 * \brief Display mode for HwSettingsWidget
 *
 * - Create: Required settings are shown as editable typed widgets (for
 *   AddProfileDialog, before the hardware object is constructed).
 * - Edit: Required settings are shown as read-only text rows (for
 *   HWDialog, where Required settings must not change post-creation).
 */
enum class HwSettingsMode { Create, Edit };

/*!
 * \brief Embeddable widget that renders hardware settings from the HardwareRegistry grouped by priority.
 *
 * Settings are drawn from the HardwareRegistry for a given hardware type and driver
 * and are presented in three tiers:
 * - \b Required — editable QFormLayout in Create mode; read-only text rows in Edit mode.
 * - \b Important — always-visible two-column table.
 * - \b Optional/Advanced — collapsible two-column table inside a QGroupBox.
 *
 * Array settings appear as a table row with an inline entry count and an Edit button
 * that opens HwArrayEditDialog.
 */
class HwSettingsWidget : public QWidget
{
    Q_OBJECT
public:
    /*!
     * \brief Construct the settings widget
     * \param hwType     Hardware type key (e.g., "FtmwDigitizer")
     * \param impl       Implementation key (e.g., "VirtualFtmwScope")
     * \param mode       Create or Edit mode (controls Required section editability)
     * \param storageKey SettingsStorage key to pre-populate current values from.
     *                   Pass an empty string in Create mode to use registry defaults.
     * \param parent     Parent widget
     */
    explicit HwSettingsWidget(const QString &hwType,
                              const QString &impl,
                              HwSettingsMode mode,
                              const QString &storageKey = {},
                              QWidget *parent = nullptr);

    /*!
     * \brief Return current scalar values keyed by SettingsStorage key.
     *
     * In Create mode returns all priority tiers. In Edit mode returns
     * Important and Optional values only (Required are read-only).
     */
    QHash<QString, QVariant> values() const;

    /*!
     * \brief Return current array values keyed by array key.
     */
    QMap<QString, std::vector<SettingsStorage::SettingsMap>> arrayValues() const;

    /*!
     * \brief Write all scalar and array values to SettingsStorage.
     * \param storageKey The SettingsStorage key for the hardware instance.
     */
    void saveToStorage(const QString &storageKey) const;

private:
    void populate(const QString &storageKey);

    QWidget *makeScalarWidget(const HwSettingDef &def, const QVariant &currentValue);
    QVariant readWidget(QWidget *widget, const QVariant &defaultValue) const;

    void addTableRow(QTableWidget *table, const QString &label,
                     const QString &description, QWidget *valueWidget);
    void addArrayTableRow(QTableWidget *table, const HwArraySettingDef &def);

    QStringList subKeysForArray(const HwArraySettingDef &def) const;

    QString d_hwType;
    QString d_impl;
    HwSettingsMode d_mode;

    QTabWidget *p_tabWidget{nullptr};
    QLabel *p_noSettingsLabel{nullptr};

    QFormLayout *p_requiredLayout{nullptr};
    QGroupBox *p_requiredGroup{nullptr};

    QTableWidget *p_importantTable{nullptr};
    QGroupBox *p_importantGroup{nullptr};

    QTableWidget *p_advancedTable{nullptr};

    // key → input widget for scalar settings
    QHash<QString, QWidget*> d_scalarWidgets;

    // array key → current entries (updated by HwArrayEditDialog on accept)
    QMap<QString, std::vector<SettingsStorage::SettingsMap>> d_arrayValues;
};

#endif // HWSETTINGSWIDGET_H
