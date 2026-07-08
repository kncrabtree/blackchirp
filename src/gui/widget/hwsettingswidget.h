#ifndef HWSETTINGSWIDGET_H
#define HWSETTINGSWIDGET_H

#include <QWidget>
#include <QHash>
#include <QMap>
#include <vector>
#include <data/storage/settingsstorage.h>
#include <data/lif/lifunits.h>
#include <hardware/core/hardwareregistry.h>

class QFormLayout;
class QGroupBox;
class QLabel;
class QTabWidget;
class QComboBox;
class ScientificSpinBox;
class SettingsTable;

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
     * \param impl       Implementation key (e.g., "VirtualFtmwDigitizer")
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
    /*!
     * \brief Bookkeeping for one HwSettingDef::displayUnitKey-linked scalar
     *        box: which double box, which sibling unit combo, and the
     *        display unit it is currently configured for (needed so a unit
     *        change can convert from the *previous* display value rather
     *        than re-deriving from the stale registered cm⁻¹ bounds).
     */
    struct UnitLinkedScalar {
        ScientificSpinBox *box;    ///< The double-typed setting's widget (makeScalarWidget always uses ScientificSpinBox for QMetaType::Double defs; it is a QAbstractSpinBox, not a QDoubleSpinBox).
        QComboBox *unitCombo;      ///< Sibling LaserUnit combo box (EnumComboBoxBase).
        QString settingKey;        ///< def.key for the linked double box.
        BC::LifConv::LaserUnit displayedUnit; ///< Unit the box is currently showing.
        QVariant minCm1;           ///< Registered def.minimum (canonical cm⁻¹), may be invalid.
        QVariant maxCm1;           ///< Registered def.maximum (canonical cm⁻¹), may be invalid.
    };

    void populate(const QString &storageKey);

    QWidget *makeScalarWidget(const HwSettingDef &def, const QVariant &currentValue);
    QVariant readWidget(QWidget *widget, const QVariant &defaultValue) const;
    QVariant scalarValueForStorage(const HwSettingDef &def) const;

    void addArrayTableRow(SettingsTable *table, const HwArraySettingDef &def);

    QStringList subKeysForArray(const HwArraySettingDef &def) const;

    /*!
     * \brief Link the display-unit-aware scalar boxes registered via
     *        HwSettingDef::displayUnitKey to their sibling LaserUnit combo
     *        boxes, converting the box's registered cm⁻¹ range/value to the
     *        combo's currently-selected display unit.
     *
     * Called once after the scalar-widget loop in populate() so build order
     * within d_scalarWidgets does not matter. Only settings whose
     * displayUnitKey names a sibling widget that resolves to a
     * BC::LifConv::LaserUnit are linked; anything else is left as a plain
     * cm⁻¹ box.
     */
    void linkDisplayUnitScalars(const QVector<HwSettingDef> &settingDefs);

    /*!
     * \brief Reconfigure a display-unit-linked box (range, suffix, decimals,
     *        value) for display unit \a u, converting the caller-supplied
     *        canonical cm⁻¹ value \a canonicalValue into the new unit.
     */
    void applyDisplayUnit(UnitLinkedScalar &linked, BC::LifConv::LaserUnit u,
                          double canonicalValue);

    QString d_hwType;
    QString d_impl;
    HwSettingsMode d_mode;

    QTabWidget *p_tabWidget{nullptr};
    QLabel *p_noSettingsLabel{nullptr};

    QFormLayout *p_requiredLayout{nullptr};
    QGroupBox *p_requiredGroup{nullptr};

    SettingsTable *p_importantTable{nullptr};
    QGroupBox *p_importantGroup{nullptr};

    SettingsTable *p_advancedTable{nullptr};

    // key → input widget for scalar settings
    QHash<QString, QWidget*> d_scalarWidgets;

    // array key → current entries (updated by HwArrayEditDialog on accept)
    QMap<QString, std::vector<SettingsStorage::SettingsMap>> d_arrayValues;

    // display-unit-linked scalar boxes (HwSettingDef::displayUnitKey), keyed
    // implicitly by settingKey — see linkDisplayUnitScalars()
    std::vector<UnitLinkedScalar> d_unitLinkedScalars;
};

#endif // HWSETTINGSWIDGET_H
