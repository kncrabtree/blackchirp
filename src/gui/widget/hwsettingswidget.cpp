#include "hwsettingswidget.h"

#include <QFormLayout>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QTabWidget>
#include <QScrollArea>
#include <QLabel>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QLineEdit>
#include <QPushButton>
#include <limits>

#include <gui/widget/scientificspinbox.h>
#include <gui/widget/settingstable.h>
#include <gui/dialog/hwarrayeditdialog.h>
#include <data/storage/settingsstorage.h>

namespace {

// The native settings key is appended so users writing Python hardware
// drivers can read off the string to pass to self.settings.get / .set.
template <class Def>
QString settingTooltip(const Def &def)
{
    return def.description + "\nKey: "_L1 + def.key;
}

} // namespace

// ---------------------------------------------------------------------------

HwSettingsWidget::HwSettingsWidget(const QString &hwType,
                                   const QString &impl,
                                   HwSettingsMode mode,
                                   const QString &storageKey,
                                   QWidget *parent)
    : QWidget(parent), d_hwType(hwType), d_impl(impl), d_mode(mode)
{
    auto *vbl = new QVBoxLayout(this);
    vbl->setContentsMargins(0, 0, 0, 0);

    p_tabWidget = new QTabWidget(this);
    p_tabWidget->setMinimumHeight(200);
    vbl->addWidget(p_tabWidget);

    // ---- "Settings" tab: Required + Important ----
    auto *settingsContent = new QWidget();
    auto *settingsVbl = new QVBoxLayout(settingsContent);
    settingsVbl->setContentsMargins(4, 4, 4, 4);

    p_requiredGroup = new QGroupBox("Required Settings", settingsContent);
    p_requiredLayout = new QFormLayout(p_requiredGroup);
    p_requiredGroup->hide();
    settingsVbl->addWidget(p_requiredGroup);

    p_importantGroup = new QGroupBox("Important Settings", settingsContent);
    auto *importantVbl = new QVBoxLayout(p_importantGroup);
    importantVbl->setContentsMargins(4, 4, 4, 4);
    p_importantTable = new SettingsTable(settingsContent);
    importantVbl->addWidget(p_importantTable);
    p_importantGroup->hide();
    settingsVbl->addWidget(p_importantGroup);
    settingsVbl->addStretch(1);

    auto *settingsScrollArea = new QScrollArea();
    settingsScrollArea->setWidgetResizable(true);
    settingsScrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    settingsScrollArea->setFrameShape(QFrame::NoFrame);
    settingsScrollArea->setWidget(settingsContent);
    p_tabWidget->addTab(settingsScrollArea, tr("Settings"));

    // Advanced table: created here, wired into a tab in populate() if needed.
    // Hidden until populate() places it inside a QScrollArea tab.
    p_advancedTable = new SettingsTable(this);
    p_advancedTable->hide();

    // Shown in place of the tab widget when there are no settings to display
    p_noSettingsLabel = new QLabel(tr("No settings available."), this);
    p_noSettingsLabel->setAlignment(Qt::AlignCenter);
    p_noSettingsLabel->hide();
    vbl->addWidget(p_noSettingsLabel);

    populate(storageKey);
}

// ---------------------------------------------------------------------------

void HwSettingsWidget::populate(const QString &storageKey)
{
    auto &reg = HardwareRegistry::instance();
    auto settingDefs = reg.getSettingDefs(d_hwType, d_impl);
    auto arrayDefs = reg.getArraySettingDefs(d_hwType, d_impl);

    // Optionally load current values from SettingsStorage
    std::unique_ptr<SettingsStorage> storage;
    if (!storageKey.isEmpty())
        storage = std::make_unique<SettingsStorage>(storageKey, SettingsStorage::Hardware);

    auto currentValue = [&](const HwSettingDef &def) -> QVariant {
        if (storage) {
            auto v = storage->get(def.key, QVariant{});
            if (v.isValid())
                return v;
        }
        return def.defaultValue;
    };

    // ---- Scalar settings ----
    bool hasRequired = false, hasImportant = false, hasAdvanced = false;

    for (const auto &def : settingDefs) {
        QVariant val = currentValue(def);

        const QString tooltip = settingTooltip(def);
        switch (def.priority) {
        case HwSettingPriority::Required:
            if (d_mode == HwSettingsMode::Create) {
                auto *w = makeScalarWidget(def, val);
                if (w) {
                    w->setToolTip(tooltip);
                    d_scalarWidgets[def.key] = w;
                    p_requiredLayout->addRow(def.label + ":", w);
                }
            } else {
                // Edit mode: read-only text
                auto *lbl = new QLabel(val.toString(), this);
                lbl->setToolTip(tooltip);
                p_requiredLayout->addRow(def.label + ":", lbl);
            }
            hasRequired = true;
            break;

        case HwSettingPriority::Important: {
            auto *w = makeScalarWidget(def, val);
            if (w) {
                w->setToolTip(tooltip);
                d_scalarWidgets[def.key] = w;
                p_importantTable->addSettingRow(def.label, w, tooltip);
            }
            hasImportant = true;
            break;
        }

        case HwSettingPriority::Optional: {
            auto *w = makeScalarWidget(def, val);
            if (w) {
                w->setToolTip(tooltip);
                d_scalarWidgets[def.key] = w;
                p_advancedTable->addSettingRow(def.label, w, tooltip);
            }
            hasAdvanced = true;
            break;
        }
        }
    }

    // ---- Array settings ----
    for (auto it = arrayDefs.cbegin(); it != arrayDefs.cend(); ++it) {
        const auto &def = it.value();

        // Load current entries from storage if available, else use registered defaults
        std::vector<SettingsStorage::SettingsMap> entries = def.entries;
        if (storage) {
            auto stored = storage->getArray(def.key);
            if (!stored.empty())
                entries = stored;
        }
        d_arrayValues[def.key] = entries;

        switch (def.priority) {
        case HwSettingPriority::Required:
            // Required arrays: show count in the Required form section
            {
                QString summary = QString("%1 entries").arg(entries.size());
                auto *lbl = new QLabel(summary, this);
                lbl->setToolTip(settingTooltip(def));
                if (d_mode == HwSettingsMode::Create) {
                    // Add an Edit button alongside the summary
                    auto *container = new QWidget(this);
                    auto *hbl = new QHBoxLayout(container);
                    hbl->setContentsMargins(0, 0, 0, 0);
                    hbl->addWidget(lbl);
                    auto *btn = new QPushButton("Edit...", container);
                    hbl->addWidget(btn);
                    hbl->addStretch(1);
                    connect(btn, &QPushButton::clicked, this, [this, def, lbl]() mutable {
                        auto subKeys = subKeysForArray(def);
                        auto *dlg = new HwArrayEditDialog(def.label, subKeys,
                                                          d_arrayValues[def.key], this);
                        if (dlg->exec() == QDialog::Accepted) {
                            d_arrayValues[def.key] = dlg->result();
                            lbl->setText(QString("%1 entries").arg(d_arrayValues[def.key].size()));
                        }
                    });
                    p_requiredLayout->addRow(def.label + ":", container);
                } else {
                    p_requiredLayout->addRow(def.label + ":", lbl);
                }
                hasRequired = true;
            }
            break;

        case HwSettingPriority::Important:
            addArrayTableRow(p_importantTable, def);
            hasImportant = true;
            break;

        case HwSettingPriority::Optional:
            addArrayTableRow(p_advancedTable, def);
            hasAdvanced = true;
            break;
        }
    }

    // Show/hide sections within the Settings tab
    p_requiredGroup->setVisible(hasRequired);
    p_importantGroup->setVisible(hasImportant);

    // Add "Advanced" tab only when there are advanced settings
    if (hasAdvanced) {
        auto *advancedScrollArea = new QScrollArea();
        advancedScrollArea->setWidgetResizable(true);
        advancedScrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        advancedScrollArea->setFrameShape(QFrame::NoFrame);
        advancedScrollArea->setWidget(p_advancedTable);
        p_tabWidget->addTab(advancedScrollArea, tr("Advanced"));
    }

    // If no settings exist at all, hide the tab widget and show a placeholder
    bool hasAny = hasRequired || hasImportant || hasAdvanced;
    p_tabWidget->setVisible(hasAny);
    p_noSettingsLabel->setVisible(!hasAny);
}

// ---------------------------------------------------------------------------

QWidget *HwSettingsWidget::makeScalarWidget(const HwSettingDef &def,
                                             const QVariant &currentValue)
{
    int typeId = def.defaultValue.userType();
    QWidget *widget = nullptr;

    if (typeId == QMetaType::Int) {
        auto *sb = new QSpinBox(this);
        sb->setRange(def.minimum.isValid() ? def.minimum.toInt() : std::numeric_limits<int>::min(),
                     def.maximum.isValid() ? def.maximum.toInt() : std::numeric_limits<int>::max());
        sb->setValue(currentValue.toInt());
        widget = sb;
    } else if (typeId == QMetaType::UInt) {
        auto *sb = new QSpinBox(this);
        sb->setRange(def.minimum.isValid() ? static_cast<int>(def.minimum.toUInt()) : 0,
                     def.maximum.isValid() ? static_cast<int>(def.maximum.toUInt())
                                           : std::numeric_limits<int>::max());
        sb->setValue(static_cast<int>(currentValue.toUInt()));
        widget = sb;
    } else if (typeId == QMetaType::Double) {
        auto *ssb = new ScientificSpinBox(this);
        if (def.minimum.isValid()) ssb->setMinimum(def.minimum.toDouble());
        if (def.maximum.isValid()) ssb->setMaximum(def.maximum.toDouble());
        ssb->setValue(currentValue.toDouble());
        widget = ssb;
    } else if (typeId == QMetaType::Bool) {
        auto *cb = new QCheckBox(this);
        cb->setChecked(currentValue.toBool());
        widget = cb;
    } else {
        auto *le = new QLineEdit(this);
        le->setText(currentValue.toString());
        widget = le;
    }

    return widget;
}

QVariant HwSettingsWidget::readWidget(QWidget *widget, const QVariant &defaultValue) const
{
    if (!widget)
        return defaultValue;

    if (auto *sb = qobject_cast<QSpinBox*>(widget))
        return sb->value();
    if (auto *ssb = qobject_cast<ScientificSpinBox*>(widget))
        return ssb->value();
    if (auto *dsb = qobject_cast<QDoubleSpinBox*>(widget))
        return dsb->value();
    if (auto *cb = qobject_cast<QCheckBox*>(widget))
        return cb->isChecked();
    if (auto *le = qobject_cast<QLineEdit*>(widget))
        return le->text();

    return defaultValue;
}

// ---------------------------------------------------------------------------

void HwSettingsWidget::addArrayTableRow(SettingsTable *table, const HwArraySettingDef &def)
{
    // Value cell: "N entries" label + "Edit..." button
    auto *countLabel = new QLabel(
        QString("%1 entries").arg(d_arrayValues.value(def.key).size()), this);

    auto *btn = new QPushButton("Edit...", this);

    // Capture by value for the key; def ref would dangle
    const QString arrayKey = def.key;
    const QString arrayLabel = def.label;
    connect(btn, &QPushButton::clicked, this, [this, arrayKey, arrayLabel, countLabel]() {
        auto &reg = HardwareRegistry::instance();
        auto arrayDefs = reg.getArraySettingDefs(d_hwType, d_impl);
        auto it = arrayDefs.find(arrayKey);
        QStringList subKeys;
        if (it != arrayDefs.end())
            subKeys = subKeysForArray(it.value());
        else if (!d_arrayValues[arrayKey].empty())
            for (auto const &[k, v] : d_arrayValues[arrayKey].front())
                subKeys.append(k);

        auto *dlg = new HwArrayEditDialog(arrayLabel, subKeys,
                                          d_arrayValues[arrayKey], this);
        if (dlg->exec() == QDialog::Accepted) {
            d_arrayValues[arrayKey] = dlg->result();
            countLabel->setText(
                QString("%1 entries").arg(d_arrayValues[arrayKey].size()));
        }
    });

    table->addSettingRow(def.label, countLabel, btn, settingTooltip(def));
}

QStringList HwSettingsWidget::subKeysForArray(const HwArraySettingDef &def) const
{
    QStringList out;
    if (!def.entries.empty()) {
        for (auto const &[k, v] : def.entries.front())
            out.append(k);
    } else {
        auto it = d_arrayValues.constFind(def.key);
        if (it != d_arrayValues.cend() && !it->empty())
            for (auto const &[k, v] : it->front())
                out.append(k);
    }
    return out;
}

// ---------------------------------------------------------------------------

QHash<QString, QVariant> HwSettingsWidget::values() const
{
    QHash<QString, QVariant> out;
    auto &reg = HardwareRegistry::instance();
    auto defs = reg.getSettingDefs(d_hwType, d_impl);

    for (const auto &def : defs) {
        // In Edit mode, Required settings are shown read-only and are not
        // in d_scalarWidgets — leave them untouched in storage.
        auto it = d_scalarWidgets.find(def.key);
        if (it != d_scalarWidgets.end())
            out[def.key] = readWidget(it.value(), def.defaultValue);
    }
    return out;
}

QMap<QString, std::vector<SettingsStorage::SettingsMap>> HwSettingsWidget::arrayValues() const
{
    return d_arrayValues;
}

void HwSettingsWidget::saveToStorage(const QString &storageKey) const
{
    SettingsStorage storage(storageKey, SettingsStorage::Hardware);

    // Scalar settings
    for (auto it = d_scalarWidgets.cbegin(); it != d_scalarWidgets.cend(); ++it) {
        // Find the default value for this key from the registry
        QVariant defaultVal;
        auto &reg = HardwareRegistry::instance();
        for (const auto &def : reg.getSettingDefs(d_hwType, d_impl)) {
            if (def.key == it.key()) {
                defaultVal = def.defaultValue;
                break;
            }
        }
        storage.set(it.key(), readWidget(it.value(), defaultVal));
    }

    // Array settings
    for (auto it = d_arrayValues.cbegin(); it != d_arrayValues.cend(); ++it)
        storage.setArray(it.key(), it.value());

    storage.save();
}
