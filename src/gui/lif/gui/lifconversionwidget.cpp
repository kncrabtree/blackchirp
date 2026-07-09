#include <gui/lif/gui/lifconversionwidget.h>

#include <QComboBox>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QInputDialog>
#include <QLabel>
#include <QLineEdit>
#include <QMenu>
#include <QMessageBox>
#include <QPushButton>
#include <QTableView>
#include <QVBoxLayout>

#include <data/lif/lifconfig.h>
#include <data/loadout/loadoutmanager.h>
#include <data/storage/settingsstorage.h>
#include <gui/style/themecolors.h>
#include <hardware/core/liflaser/liflaser.h>

using namespace Qt::StringLiterals;
using namespace BC::LifConv;

LifConversionWidget::LifConversionWidget(bool showDeleteButton, QWidget *parent) :
    QWidget(parent)
{
    auto mainLayout = new QVBoxLayout(this);

    // ── Preset bar ──────────────────────────────────────────────────────────
    auto presetGroup = new QGroupBox("LIF Preset"_L1, this);
    auto presetRow = new QHBoxLayout(presetGroup);

    p_presetCombo = new QComboBox(presetGroup);
    p_presetCombo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    presetRow->addWidget(p_presetCombo, 1);

    p_applyPresetButton  = new QPushButton("Apply"_L1,      presetGroup);
    p_savePresetButton   = new QPushButton("Save"_L1,       presetGroup);
    p_saveAsPresetButton = new QPushButton("Save As..."_L1, presetGroup);
    p_renamePresetButton = new QPushButton("Rename..."_L1,  presetGroup);
    p_deletePresetButton = new QPushButton("Delete"_L1,     presetGroup);

    for(auto *btn : {p_applyPresetButton, p_savePresetButton, p_saveAsPresetButton,
                      p_renamePresetButton, p_deletePresetButton})
        presetRow->addWidget(btn);

    p_deletePresetButton->setVisible(showDeleteButton);

    d_applyIcon = ThemeColors::createThemedIconWithStates(
        ":/icons/arrow-down-on-square.svg", ThemeColors::IconPrimary, ThemeColors::DisabledText, this);
    d_resetIcon = ThemeColors::createThemedIconWithStates(
        ":/icons/arrow-path.svg", ThemeColors::IconPrimary, ThemeColors::DisabledText, this);
    p_applyPresetButton->setIcon(d_applyIcon);
    p_savePresetButton->setIcon(ThemeColors::createThemedIconWithStates(
        ":/icons/archive-box.svg", ThemeColors::IconPrimary, ThemeColors::DisabledText, this));
    p_saveAsPresetButton->setIcon(ThemeColors::createThemedIconWithStates(
        ":/icons/arrow-up-on-square.svg", ThemeColors::IconPrimary, ThemeColors::DisabledText, this));
    p_renamePresetButton->setIcon(ThemeColors::createThemedIconWithStates(
        ":/icons/pencil.svg", ThemeColors::IconPrimary, ThemeColors::DisabledText, this));
    p_deletePresetButton->setIcon(ThemeColors::createThemedIconWithStates(
        ":/icons/trash.svg", ThemeColors::StatusError, ThemeColors::DisabledText, this));

    mainLayout->addWidget(presetGroup);

    // ── Table ───────────────────────────────────────────────────────────────
    p_model = new LifConversionTableModel(this);
    p_tableView = new QTableView(this);
    p_tableView->setModel(p_model);
    // Stage and the two input columns carry the long text (hardware keys and
    // wiring descriptions), so they share the free space; op, harmonic, and the
    // FINAL marker stay just wide enough for their contents.
    auto *header = p_tableView->horizontalHeader();
    header->setStretchLastSection(false);
    header->setSectionResizeMode(QHeaderView::ResizeToContents);
    header->setSectionResizeMode(LifConversionTableModel::StageColumn, QHeaderView::Stretch);
    header->setSectionResizeMode(LifConversionTableModel::Input0Column, QHeaderView::Stretch);
    header->setSectionResizeMode(LifConversionTableModel::Input1Column, QHeaderView::Stretch);
    p_tableView->setItemDelegate(new LifConversionTableDelegate(p_tableView));
    p_tableView->setContextMenuPolicy(Qt::CustomContextMenu);
    mainLayout->addWidget(p_tableView, 1);

    // ── Preview footer ──────────────────────────────────────────────────────
    p_previewLabel = new QLabel(this);
    p_previewLabel->setWordWrap(true);
    p_previewLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    p_previewLabel->setFrameShape(QFrame::StyledPanel);
    mainLayout->addWidget(p_previewLabel);

    // ── Initial seeding (before dirty connections) ─────────────────────────
    const auto currentName = LoadoutManager::instance().currentLoadoutName();
    auto currentPreset = LoadoutManager::instance().currentLifPreset(currentName);
    if(currentPreset)
        initializeFromLifPreset(*currentPreset);

    // ── Connections ─────────────────────────────────────────────────────────
    connect(p_model, &LifConversionTableModel::edited, this, &LifConversionWidget::edited);
    connect(p_model, &LifConversionTableModel::edited, this, &LifConversionWidget::markDirty);
    connect(p_model, &LifConversionTableModel::edited, this, &LifConversionWidget::updatePreview);
    connect(p_model, &LifConversionTableModel::applyHarmonic, this, &LifConversionWidget::applyHarmonic);

    connect(p_tableView, &QTableView::customContextMenuRequested,
            this, &LifConversionWidget::showTableContextMenu);

    connect(p_presetCombo, &QComboBox::currentIndexChanged,
            this, &LifConversionWidget::updatePresetBar);
    connect(p_applyPresetButton,  &QPushButton::clicked, this, &LifConversionWidget::onApplyPreset);
    connect(p_savePresetButton,   &QPushButton::clicked, this, &LifConversionWidget::onSavePreset);
    connect(p_saveAsPresetButton, &QPushButton::clicked, this, &LifConversionWidget::onSaveAsPreset);
    connect(p_renamePresetButton, &QPushButton::clicked, this, &LifConversionWidget::onRenamePreset);
    connect(p_deletePresetButton, &QPushButton::clicked, this, &LifConversionWidget::onDeletePreset);

    auto &lm = LoadoutManager::instance();
    connect(&lm, &LoadoutManager::lifPresetAdded,
            this, [this](auto, auto) { populatePresetCombo(); });
    connect(&lm, &LoadoutManager::lifPresetRemoved,
            this, [this](auto, auto) { populatePresetCombo(); });
    connect(&lm, &LoadoutManager::lifPresetChanged,
            this, [this](auto, auto) { populatePresetCombo(); });
    connect(&lm, &LoadoutManager::currentLifPresetChanged,
            this, [this](auto, auto) { populatePresetCombo(); });
    connect(&lm, &LoadoutManager::currentLoadoutChanged,
            this, [this](auto) { populatePresetCombo(); });

    populatePresetCombo();
    updatePreview();
}

LifConversionWidget::~LifConversionWidget()
{
}

void LifConversionWidget::setFromConfig(const LifConfig &cfg)
{
    d_suppressDirty = true;
    p_model->setFromConfig(cfg);
    d_suppressDirty = false;
    updatePreview();
}

void LifConversionWidget::toConfig(LifConfig &cfg) const
{
    p_model->toConfig(cfg);
}

void LifConversionWidget::harmonicApplied(const QString &stageKey)
{
    p_model->harmonicApplied(stageKey);
}

void LifConversionWidget::initializeFromLifPreset(const LifPreset &preset)
{
    d_suppressDirty = true;
    p_model->setFromSnapshot(preset.conversion);
    d_suppressDirty = false;
}

LifPreset LifConversionWidget::toLifPreset() const
{
    LifPreset preset;
    preset.conversion = p_model->toSnapshot();
    return preset;
}

void LifConversionWidget::clearDirty()
{
    const bool changed = d_dirty;
    d_dirty = false;
    if(changed)
        emit dirtyChanged(false);
    updatePresetBar();
}

void LifConversionWidget::markDirty()
{
    if(d_suppressDirty)
        return;
    const bool changed = !d_dirty;
    d_dirty = true;
    if(changed)
        emit dirtyChanged(true);
    updatePresetBar();
}

void LifConversionWidget::populatePresetCombo()
{
    const auto activeName = LoadoutManager::instance().currentLoadoutName();
    const auto presets = LoadoutManager::instance().lifPresetNames(activeName, false);
    const auto currentPresetName = LoadoutManager::instance().currentLifPresetName(activeName);

    p_presetCombo->blockSignals(true);
    p_presetCombo->clear();
    for(const auto &name : presets)
        p_presetCombo->addItem(name);
    const int idx = p_presetCombo->findText(currentPresetName);
    p_presetCombo->setCurrentIndex(idx);
    p_presetCombo->blockSignals(false);

    updatePresetBar();
}

void LifConversionWidget::updatePresetBar()
{
    const auto activeName = LoadoutManager::instance().currentLoadoutName();
    const auto currentPresetName = LoadoutManager::instance().currentLifPresetName(activeName);
    const bool isReal = !currentPresetName.isEmpty()
        && currentPresetName != BC::Store::LM::lastUsedLifPresetName;
    const bool comboHasSelection = p_presetCombo->currentIndex() >= 0;

    const QString comboName = p_presetCombo->currentText();
    const bool comboMatchesCurrent = comboHasSelection && comboName == currentPresetName;
    if(comboMatchesCurrent)
    {
        p_applyPresetButton->setText("Reset"_L1);
        p_applyPresetButton->setIcon(d_resetIcon);
        p_applyPresetButton->setEnabled(d_dirty);
    }
    else
    {
        p_applyPresetButton->setText("Apply"_L1);
        p_applyPresetButton->setIcon(d_applyIcon);
        p_applyPresetButton->setEnabled(comboHasSelection);
    }
    p_savePresetButton->setEnabled(isReal && d_dirty);
    p_saveAsPresetButton->setEnabled(true);
    p_renamePresetButton->setEnabled(isReal);
    p_deletePresetButton->setEnabled(comboHasSelection && comboName != currentPresetName);
}

void LifConversionWidget::onApplyPreset()
{
    const QString name = p_presetCombo->currentText();
    if(name.isEmpty())
        return;

    if(d_dirty)
    {
        const auto r = QMessageBox::question(
            this, "Load LIF Preset"_L1,
            QString("Discard unsaved changes and load LIF preset \"%1\"?").arg(name),
            QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
        if(r != QMessageBox::Yes)
            return;
    }

    const auto activeName = LoadoutManager::instance().currentLoadoutName();
    auto preset = LoadoutManager::instance().getLifPreset(activeName, name);
    if(!preset)
        return;

    initializeFromLifPreset(*preset);
    LoadoutManager::instance().setCurrentLifPresetName(activeName, name);

    updatePreview();
    emit edited();
    clearDirty();
}

void LifConversionWidget::onSavePreset()
{
    const auto activeName = LoadoutManager::instance().currentLoadoutName();
    if(activeName.isEmpty())
        return;

    const auto currentPresetName = LoadoutManager::instance().currentLifPresetName(activeName);
    if(currentPresetName.isEmpty()
        || currentPresetName == BC::Store::LM::lastUsedLifPresetName)
        return;

    const auto preset = toLifPreset();
    LoadoutManager::instance().putLifPreset(activeName, currentPresetName, preset);
    LoadoutManager::instance().putLifPreset(
        activeName, BC::Store::LM::lastUsedLifPresetName, preset);
    clearDirty();
}

void LifConversionWidget::onSaveAsPreset()
{
    const auto activeName = LoadoutManager::instance().currentLoadoutName();
    if(activeName.isEmpty())
        return;

    bool ok;
    auto name = QInputDialog::getText(
        this, "Save LIF Preset As"_L1, "Preset name:"_L1,
        QLineEdit::Normal, {}, &ok).trimmed();
    if(!ok || name.isEmpty())
        return;

    if(name == BC::Store::LM::lastUsedLifPresetName)
    {
        QMessageBox::warning(this, "Invalid Name"_L1, "That preset name is reserved."_L1);
        return;
    }

    if(LoadoutManager::instance().lifPresetExists(activeName, name))
    {
        const auto r = QMessageBox::question(
            this, "Overwrite Preset"_L1,
            QString("Preset \"%1\" already exists. Overwrite?").arg(name),
            QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
        if(r != QMessageBox::Yes)
            return;
    }

    const auto preset = toLifPreset();
    LoadoutManager::instance().putLifPreset(activeName, name, preset);
    LoadoutManager::instance().putLifPreset(
        activeName, BC::Store::LM::lastUsedLifPresetName, preset);
    LoadoutManager::instance().setCurrentLifPresetName(activeName, name);
    clearDirty();
}

void LifConversionWidget::onRenamePreset()
{
    const auto activeName = LoadoutManager::instance().currentLoadoutName();
    if(activeName.isEmpty())
        return;

    const auto currentPresetName = LoadoutManager::instance().currentLifPresetName(activeName);
    if(currentPresetName.isEmpty()
        || currentPresetName == BC::Store::LM::lastUsedLifPresetName)
        return;

    bool ok;
    auto newName = QInputDialog::getText(
        this, "Rename LIF Preset"_L1, "New name:"_L1,
        QLineEdit::Normal, currentPresetName, &ok).trimmed();
    if(!ok || newName.isEmpty() || newName == currentPresetName)
        return;

    if(newName == BC::Store::LM::lastUsedLifPresetName)
    {
        QMessageBox::warning(this, "Invalid Name"_L1, "That preset name is reserved."_L1);
        return;
    }

    if(LoadoutManager::instance().lifPresetExists(activeName, newName))
    {
        QMessageBox::warning(this, "Name Exists"_L1,
            QString("A preset named \"%1\" already exists.").arg(newName));
        return;
    }

    LoadoutManager::instance().renameLifPreset(activeName, currentPresetName, newName);
}

void LifConversionWidget::onDeletePreset()
{
    const QString presetName = p_presetCombo->currentText();
    if(presetName.isEmpty())
        return;

    const auto activeName = LoadoutManager::instance().currentLoadoutName();
    if(activeName.isEmpty())
        return;

    const auto r = QMessageBox::question(
        this, "Delete LIF Preset"_L1,
        QString("Delete LIF preset \"%1\"?").arg(presetName),
        QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
    if(r != QMessageBox::Yes)
        return;

    LoadoutManager::instance().removeLifPreset(activeName, presetName);
}

void LifConversionWidget::showTableContextMenu(const QPoint &pos)
{
    auto index = p_tableView->indexAt(pos);
    if(!index.isValid())
        return;

    const int row = index.row();
    if(row < 0 || static_cast<std::size_t>(row) >= p_model->nodes().size())
        return;

    const auto &node = p_model->nodes().at(static_cast<std::size_t>(row));
    const auto stageKey = node.stageKey;
    const auto op = node.op;

    QMenu menu(this);
    auto changeHarmonicAction = menu.addAction("Change harmonic…"_L1);
    changeHarmonicAction->setEnabled(op == Op::NHG);
    changeHarmonicAction->setToolTip("Harmonic order only applies to NHG stages."_L1);

    auto chosen = menu.exec(p_tableView->viewport()->mapToGlobal(pos));
    if(chosen != changeHarmonicAction)
        return;

    const int current = node.n;

    bool ok = false;
    int n = QInputDialog::getInt(this, "Change Harmonic Order"_L1,
                                  QString("New harmonic order for \"%1\":").arg(stageKey),
                                  current, 1, 20, 1, &ok);
    if(!ok)
        return;

    p_model->requestHarmonicChange(stageKey, n);
}

void LifConversionWidget::updatePreview()
{
    QString text = buildChainExpression();
    auto result = p_model->assemblyResult();

    if(!result.ok)
    {
        text += u"\nValidation error: %1"_s.arg(result.errorString);
    }
    else
    {
        const auto laserKey = p_model->currentLaserKey();
        if(!laserKey.isEmpty())
        {
            SettingsStorage s(laserKey, SettingsStorage::Hardware);
            const auto lo = s.get(BC::Key::LifLaser::minPos, 5000.0);
            const auto hi = s.get(BC::Key::LifLaser::maxPos, 40000.0);
            const auto [outLo, outHi] = result.conversion.outputRange(lo, hi);
            text += u"\nOutput range: %1 – %2 cm⁻¹"_s.arg(outLo,0,'f',3).arg(outHi,0,'f',3);
        }
    }

    const auto dropped = p_model->droppedStages();
    if(!dropped.isEmpty())
        text += u"\nDropped stages (no longer active): %1"_s.arg(dropped.join(", "_L1));

    p_previewLabel->setText(text);
}

QString LifConversionWidget::buildChainExpression() const
{
    const auto &nodes = p_model->nodes();
    if(nodes.empty())
        return "Laser → FINAL (identity, no conversion stages)"_L1;

    auto describeRef = [](const InputRef &ref) -> QString {
        switch(ref.type)
        {
        case RefType::Laser:
            return "Laser"_L1;
        case RefType::Stage:
            return ref.stageKey;
        case RefType::Fixed:
            return u"%1 cm⁻¹"_s.arg(ref.fixedCm1,0,'f',3);
        }
        return {};
    };

    QStringList parts;
    for(const auto &n : nodes)
    {
        QString opLabel = n.op == Op::NHG ? u"NHG ×%1"_s.arg(n.n)
                         : n.op == Op::SFG ? u"SFG"_s : u"DFG"_s;
        QStringList ins;
        for(const auto &ref : n.inputs)
            ins << describeRef(ref);

        QString part = u"%1 = %2(%3)"_s.arg(n.stageKey, opLabel, ins.join(", "_L1));
        if(n.isFinal)
            part += " [FINAL]"_L1;
        parts << part;
    }

    return parts.join("; "_L1);
}
