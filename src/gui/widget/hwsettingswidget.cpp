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
#include <QComboBox>
#include <QPushButton>
#include <QMetaEnum>
#include <QSignalBlocker>
#include <limits>

#include <gui/widget/scientificspinbox.h>
#include <gui/widget/settingstable.h>
#include <gui/widget/enumcombobox.h>
#include <gui/dialog/hwarrayeditdialog.h>
#include <data/storage/settingsstorage.h>
#include <data/storage/enumcsvconvert.h>
#include <data/lif/lifunits.h>

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
                    pushGatedRow(def.gateKey, def.gateValue,
                                 [this, w](bool visible) { p_requiredLayout->setRowVisible(w, visible); });
                }
            } else {
                // Edit mode: read-only text
                auto *lbl = new QLabel(val.toString(), this);
                lbl->setToolTip(tooltip);
                p_requiredLayout->addRow(def.label + ":", lbl);
                pushGatedRow(def.gateKey, def.gateValue,
                             [this, lbl](bool visible) { p_requiredLayout->setRowVisible(lbl, visible); });
            }
            hasRequired = true;
            break;

        case HwSettingPriority::Important: {
            auto *w = makeScalarWidget(def, val);
            if (w) {
                w->setToolTip(tooltip);
                d_scalarWidgets[def.key] = w;
                int row = p_importantTable->addSettingRow(def.label, w, tooltip);
                pushGatedRow(def.gateKey, def.gateValue,
                             [this, row](bool visible) { p_importantTable->setRowHidden(row, !visible); });
            }
            hasImportant = true;
            break;
        }

        case HwSettingPriority::Optional: {
            auto *w = makeScalarWidget(def, val);
            if (w) {
                w->setToolTip(tooltip);
                d_scalarWidgets[def.key] = w;
                int row = p_advancedTable->addSettingRow(def.label, w, tooltip);
                pushGatedRow(def.gateKey, def.gateValue,
                             [this, row](bool visible) { p_advancedTable->setRowHidden(row, !visible); });
            }
            hasAdvanced = true;
            break;
        }
        }
    }

    // Link display-unit-aware scalar boxes (HwSettingDef::displayUnitKey) to
    // their sibling LaserUnit combo boxes. A post-pass rather than inline in
    // the loop above because build order within d_scalarWidgets is not
    // guaranteed — a box may be built before the combo box it depends on.
    linkDisplayUnitScalars(settingDefs);

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
                    pushGatedRow(def.gateKey, def.gateValue,
                                 [this, container](bool visible) { p_requiredLayout->setRowVisible(container, visible); });
                } else {
                    p_requiredLayout->addRow(def.label + ":", lbl);
                    pushGatedRow(def.gateKey, def.gateValue,
                                 [this, lbl](bool visible) { p_requiredLayout->setRowVisible(lbl, visible); });
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

    // Resolve every def/array-def with a non-empty gateKey (recorded above
    // via pushGatedRow) against its sibling gate combo and wire it live. A
    // post-pass, like linkDisplayUnitScalars(), since a gated row's gate
    // widget may be built after the row itself within d_scalarWidgets.
    applyGates();

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
    } else if (QMetaType mt = def.defaultValue.metaType(); mt.flags() & QMetaType::IsEnumeration) {
        // Q_ENUM/Q_ENUM_NS setting (e.g. BC::LifConv::LaserUnit): render a
        // combobox of the enum's keys via the shared EnumComboBoxBase, whose
        // item data is the key-name string so the persisted form matches
        // BC::CSV::enumFromVariant's read side.
        auto me = BC::CSV::metaEnumFromType(mt);
        auto *combo = new EnumComboBoxBase(me, this);

        const QString keyName = (currentValue.metaType() == mt)
            ? QString::fromUtf8(me.valueToKey(currentValue.toInt()))
            : currentValue.toString();
        combo->setCurrentKey(keyName);

        widget = combo;
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
    if (auto *combo = qobject_cast<QComboBox*>(widget))
        return combo->currentData();
    if (auto *le = qobject_cast<QLineEdit*>(widget))
        return le->text();

    return defaultValue;
}

QVariant HwSettingsWidget::scalarValueForStorage(const HwSettingDef &def) const
{
    auto it = d_scalarWidgets.constFind(def.key);
    if (it == d_scalarWidgets.cend())
        return def.defaultValue;

    if (!def.displayUnitKey.isEmpty()) {
        for (const auto &linked : d_unitLinkedScalars) {
            if (linked.settingKey == def.key)
                return BC::LifConv::toCm1(linked.box->value(), linked.displayedUnit);
        }
    }

    return readWidget(it.value(), def.defaultValue);
}

// ---------------------------------------------------------------------------

void HwSettingsWidget::linkDisplayUnitScalars(const QVector<HwSettingDef> &settingDefs)
{
    using BC::LifConv::LaserUnit;

    auto laserUnitMeta = QMetaEnum::fromType<LaserUnit>();

    for (const auto &def : settingDefs) {
        if (def.displayUnitKey.isEmpty())
            continue;

        auto boxIt = d_scalarWidgets.constFind(def.key);
        auto comboIt = d_scalarWidgets.constFind(def.displayUnitKey);
        if (boxIt == d_scalarWidgets.cend() || comboIt == d_scalarWidgets.cend())
            continue;

        // makeScalarWidget() always renders a QMetaType::Double setting as a
        // ScientificSpinBox (a QAbstractSpinBox, not a QDoubleSpinBox).
        auto *box = qobject_cast<ScientificSpinBox*>(boxIt.value());
        auto *combo = qobject_cast<QComboBox*>(comboIt.value());
        if (!box || !combo)
            continue;

        // The combo stores the enum's key-name string as item data
        // (EnumComboBoxBase); resolve it against LaserUnit specifically so a
        // displayUnitKey pointing at some other enum type is left alone.
        bool ok = false;
        int val = laserUnitMeta.keyToValue(combo->currentData().toString().toUtf8().constData(), &ok);
        if (!ok)
            continue;

        UnitLinkedScalar linked;
        linked.box = box;
        linked.unitCombo = combo;
        linked.settingKey = def.key;
        linked.displayedUnit = static_cast<LaserUnit>(val);
        linked.minCm1 = def.minimum;
        linked.maxCm1 = def.maximum;

        // box->value() is still the raw registered/stored cm⁻¹ value here —
        // makeScalarWidget() populated it before this post-pass runs.
        applyDisplayUnit(linked, linked.displayedUnit, box->value());

        d_unitLinkedScalars.push_back(linked);
    }

    // Wire live reconversion once the vector's final size for this widget is
    // known, so the index-captured lambdas below never see a reallocation
    // from a later push_back.
    for (std::size_t i = 0; i < d_unitLinkedScalars.size(); ++i) {
        auto *combo = d_unitLinkedScalars[i].unitCombo;
        connect(combo, &QComboBox::currentIndexChanged, this, [this, i]() {
            auto &linked = d_unitLinkedScalars[i];

            bool ok = false;
            auto me = QMetaEnum::fromType<LaserUnit>();
            int val = me.keyToValue(linked.unitCombo->currentData().toString().toUtf8().constData(), &ok);
            if (!ok)
                return;

            auto nu = static_cast<LaserUnit>(val);
            if (nu == linked.displayedUnit)
                return;

            // Preserve the physical value across the unit switch: derive
            // canonical cm⁻¹ from what the box currently shows in the unit
            // it was *previously* configured for, then redisplay in the new
            // unit.
            double canon = BC::LifConv::toCm1(linked.box->value(), linked.displayedUnit);
            applyDisplayUnit(linked, nu, canon);
        });
    }
}

void HwSettingsWidget::applyDisplayUnit(UnitLinkedScalar &linked, BC::LifConv::LaserUnit u,
                                        double canonicalValue)
{
    using BC::LifConv::fromCm1;

    // Convert the registered cm⁻¹ bounds to the new display unit; a
    // reciprocal unit (e.g. nm) inverts min/max order, so sort ascending.
    // Only apply a bound whose registered cm⁻¹ counterpart is valid —
    // otherwise leave that side of the box's existing (wide default) range
    // untouched.
    double lo = linked.box->minimum();
    double hi = linked.box->maximum();
    if (linked.minCm1.isValid() && linked.maxCm1.isValid()) {
        double a = fromCm1(linked.minCm1.toDouble(), u);
        double b = fromCm1(linked.maxCm1.toDouble(), u);
        lo = qMin(a, b);
        hi = qMax(a, b);
    } else if (linked.minCm1.isValid()) {
        lo = fromCm1(linked.minCm1.toDouble(), u);
    } else if (linked.maxCm1.isValid()) {
        hi = fromCm1(linked.maxCm1.toDouble(), u);
    }

    const QSignalBlocker blocker(linked.box);
    // Bump precision so sub-nm entry is possible; ScientificSpinBox exposes
    // this as displayPrecision (it is not a QDoubleSpinBox).
    linked.box->setDisplayPrecision(4);
    linked.box->setRange(lo, hi);
    linked.box->setSuffix(u" "_s + BC::LifConv::unitLabel(u));
    linked.box->setValue(fromCm1(canonicalValue, u));
    linked.displayedUnit = u;
}

// ---------------------------------------------------------------------------

void HwSettingsWidget::pushGatedRow(const QString &gateKey, const QVariant &gateValue,
                                    std::function<void(bool)> setVisible)
{
    if (gateKey.isEmpty())
        return;

    d_gatedRows.push_back({gateKey, gateValue, std::move(setVisible)});
}

void HwSettingsWidget::applyGates()
{
    // Index-based iteration (rather than a range-for capturing a reference)
    // so the connected lambdas below stay valid even though d_gatedRows is
    // a member vector — matches the d_unitLinkedScalars precedent in
    // linkDisplayUnitScalars().
    for (std::size_t i = 0; i < d_gatedRows.size(); ++i) {
        const auto &gr = d_gatedRows[i];

        auto comboIt = d_scalarWidgets.constFind(gr.gateKey);
        if (comboIt == d_scalarWidgets.cend())
            continue; // gateKey names an absent widget; leave the row visible

        // The gate widget is always the EnumComboBoxBase built by
        // makeScalarWidget() for a Q_ENUM/Q_ENUM_NS setting; its item data
        // is the enum's key-name string (see makeScalarWidget()). Cast to
        // the plain QComboBox base, as linkDisplayUnitScalars() does, since
        // only QComboBox::currentData() is needed here.
        auto *combo = qobject_cast<QComboBox*>(comboIt.value());
        if (!combo)
            continue; // gateKey names a non-enum widget; leave the row visible

        auto apply = [this, i, combo]() {
            const auto &g = d_gatedRows[i];
            // Resolve gateValue (a Q_ENUM/Q_ENUM_NS-typed QVariant) to its
            // key-name string so it compares against the combo's
            // currentData() on equal footing, mirroring how
            // linkDisplayUnitScalars() resolves LaserUnit by name rather
            // than by underlying int value.
            const bool visible =
                BC::CSV::enumKeyName(g.gateValue).toString() == combo->currentData().toString();
            g.setVisible(visible);
        };

        apply();
        connect(combo, &QComboBox::currentIndexChanged, this, apply);
    }
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

    int row = table->addSettingRow(def.label, countLabel, btn, settingTooltip(def));
    pushGatedRow(def.gateKey, def.gateValue,
                 [table, row](bool visible) { table->setRowHidden(row, !visible); });
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
            out[def.key] = scalarValueForStorage(def);
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

    // Scalar settings. Iterating the registry defs (rather than
    // d_scalarWidgets directly) lets scalarValueForStorage() see each
    // def's displayUnitKey without a second per-key registry lookup, and
    // keeps this path and values() from being able to diverge.
    auto &reg = HardwareRegistry::instance();
    for (const auto &def : reg.getSettingDefs(d_hwType, d_impl)) {
        if (d_scalarWidgets.constFind(def.key) == d_scalarWidgets.cend())
            continue;
        storage.set(def.key, scalarValueForStorage(def));
    }

    // Array settings
    for (auto it = d_arrayValues.cbegin(); it != d_arrayValues.cend(); ++it)
        storage.setArray(it.key(), it.value());

    storage.save();
}
