#include "ftmwconfigwidget.h"

#include <QComboBox>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QInputDialog>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QTabWidget>
#include <QVBoxLayout>

#include <data/bcglobals.h>
#include <data/experiment/ftmwconfig.h>
#include <data/settings/guikeys.h>
#include <data/settings/hardwarekeys.h>
#include <gui/style/themecolors.h>
#include <data/loadout/loadoutmanager.h>
#include <data/loadout/rfconfigsnapshot.h>

#include <gui/widget/chirpconfigwidget.h>
#include <gui/widget/ftmwdigitizerconfigwidget.h>
#include <gui/widget/rfconfigwidget.h>

using namespace Qt::StringLiterals;

FtmwConfigWidget::FtmwConfigWidget(const QString &awgHwKey, const QString &digiHwKey,
                                   const QHash<RfConfig::ClockType, RfConfig::ClockFreq> &currentClocks,
                                   bool showDeleteButton,
                                   QWidget *parent)
    : QWidget(parent), SettingsStorage(BC::Key::FtmwConfigWidget::key),
      d_awgHwKey(awgHwKey), d_digiHwKey(digiHwKey)
{
    auto *mainLayout = new QVBoxLayout(this);

    // ── Preset bar ──────────────────────────────────────────────────────────
    auto *presetGroup = new QGroupBox("FTMW Preset"_L1, this);
    auto *presetRow = new QHBoxLayout(presetGroup);

    p_ftmwPresetCombo = new QComboBox(presetGroup);
    p_ftmwPresetCombo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    presetRow->addWidget(p_ftmwPresetCombo, 1);

    p_presetStatusLabel = new QLabel(presetGroup);
    p_presetStatusLabel->setVisible(false);
    presetRow->addWidget(p_presetStatusLabel);

    p_applyPresetButton      = new QPushButton("Apply"_L1,        presetGroup);
    p_savePresetButton       = new QPushButton("Save"_L1,         presetGroup);
    p_saveAsPresetButton     = new QPushButton("Save As..."_L1,   presetGroup);
    p_renamePresetButton     = new QPushButton("Rename..."_L1,    presetGroup);
    p_deletePresetButton     = new QPushButton("Delete"_L1,       presetGroup);
    p_setDefaultPresetButton = new QPushButton("Set Default"_L1,  presetGroup);

    for (auto *btn : {p_applyPresetButton, p_savePresetButton, p_saveAsPresetButton,
                      p_renamePresetButton, p_deletePresetButton, p_setDefaultPresetButton})
        presetRow->addWidget(btn);

    p_deletePresetButton->setVisible(showDeleteButton);

    mainLayout->addWidget(presetGroup);

    // ── Tabs ─────────────────────────────────────────────────────────────────
    p_tabWidget = new QTabWidget(this);

    // RF tab
    auto *rfTab = new QWidget;
    auto *rfTabLayout = new QVBoxLayout(rfTab);
    auto *rfSourceRow = new QHBoxLayout;
    rfSourceRow->addWidget(new QLabel("Copy from other FTMW Preset:"_L1, rfTab));
    p_rfSourceCombo = new QComboBox(rfTab);
    p_rfSourceCombo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    rfSourceRow->addWidget(p_rfSourceCombo, 1);
    rfTabLayout->addLayout(rfSourceRow);
    p_rfWidget = new RfConfigWidget(rfTab);
    rfTabLayout->addWidget(p_rfWidget, 1);
    p_tabWidget->addTab(rfTab, "RF Config"_L1);

    // Chirp tab
    auto *chirpTab = new QWidget;
    auto *chirpTabLayout = new QVBoxLayout(chirpTab);
    auto *chirpSourceRow = new QHBoxLayout;
    chirpSourceRow->addWidget(new QLabel("Copy from other FTMW Preset:"_L1, chirpTab));
    p_chirpSourceCombo = new QComboBox(chirpTab);
    p_chirpSourceCombo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    chirpSourceRow->addWidget(p_chirpSourceCombo, 1);
    chirpTabLayout->addLayout(chirpSourceRow);
    p_chirpWidget = new ChirpConfigWidget(chirpTab);
    chirpTabLayout->addWidget(p_chirpWidget, 1);
    p_tabWidget->addTab(chirpTab, "Chirp Config"_L1);

    // Digitizer tab
    auto *digiTab = new QWidget;
    auto *digiTabLayout = new QVBoxLayout(digiTab);
    auto *digiSourceRow = new QHBoxLayout;
    digiSourceRow->addWidget(new QLabel("Copy from other FTMW Preset:"_L1, digiTab));
    p_digiSourceCombo = new QComboBox(digiTab);
    p_digiSourceCombo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    digiSourceRow->addWidget(p_digiSourceCombo, 1);
    digiTabLayout->addLayout(digiSourceRow);
    p_digiWidget = new FtmwDigitizerConfigWidget(digiTab);
    p_digiWidget->d_maxAnalogEnabled = 1;
    digiTabLayout->addWidget(p_digiWidget, 1);
    p_tabWidget->addTab(digiTab, "Digitizer Config"_L1);

    mainLayout->addWidget(p_tabWidget, 1);

    // ── Initial seeding (before dirty connections) ────────────────────────
    const auto currentName = LoadoutManager::instance().currentLoadoutName();
    auto currentPreset = LoadoutManager::instance().currentFtmwPreset(currentName);
    if (currentPreset) {
        initializeFromFtmwPreset(*currentPreset);
    } else {
        const auto defName = LoadoutManager::instance().defaultFtmwPresetName(currentName);
        auto defaultPreset = defName.isEmpty()
            ? std::optional<FtmwPreset>{}
            : LoadoutManager::instance().getFtmwPreset(currentName, defName);
        if (defaultPreset) {
            initializeFromFtmwPreset(*defaultPreset);
        } else {
            RfConfig rfc;
            rfc.setCurrentClocks(currentClocks);
            p_rfWidget->setClocks(currentClocks);
            p_chirpWidget->initialize(rfc);
        }
    }

    // ── Connections ──────────────────────────────────────────────────────────
    connect(p_rfWidget, &RfConfigWidget::applyClocks, this, &FtmwConfigWidget::applyClocks);

    populateSourceCombos();

    connect(p_rfSourceCombo, &QComboBox::currentIndexChanged,
            this, &FtmwConfigWidget::onRfSourceChanged);
    connect(p_chirpSourceCombo, &QComboBox::currentIndexChanged,
            this, &FtmwConfigWidget::onChirpSourceChanged);
    connect(p_digiSourceCombo, &QComboBox::currentIndexChanged,
            this, &FtmwConfigWidget::onDigiSourceChanged);

    connect(p_tabWidget, &QTabWidget::currentChanged, this, [this](int index) {
        if (index == 1) {
            RfConfig rfc;
            p_rfWidget->toRfConfig(rfc);
            d_suppressDirty = true;
            p_chirpWidget->initialize(rfc);
            p_chirpWidget->updateChirpPlot();
            d_suppressDirty = false;
        }
    });

    // Dirty tracking (after seeding so initial values don't mark widget dirty)
    connect(p_rfWidget, &RfConfigWidget::edited,
            this, &FtmwConfigWidget::markDirty);
    connect(p_chirpWidget, &ChirpConfigWidget::chirpConfigChanged,
            this, &FtmwConfigWidget::markDirty);
    connect(p_digiWidget, &FtmwDigitizerConfigWidget::edited,
            this, &FtmwConfigWidget::markDirty);

    // Preset bar connections
    connect(p_ftmwPresetCombo, &QComboBox::currentIndexChanged,
            this, &FtmwConfigWidget::updatePresetBar);
    connect(p_applyPresetButton,      &QPushButton::clicked, this, &FtmwConfigWidget::onApplyPreset);
    connect(p_savePresetButton,       &QPushButton::clicked, this, &FtmwConfigWidget::onSavePreset);
    connect(p_saveAsPresetButton,     &QPushButton::clicked, this, &FtmwConfigWidget::onSaveAsPreset);
    connect(p_renamePresetButton,     &QPushButton::clicked, this, &FtmwConfigWidget::onRenamePreset);
    connect(p_deletePresetButton,     &QPushButton::clicked, this, &FtmwConfigWidget::onDeletePreset);
    connect(p_setDefaultPresetButton, &QPushButton::clicked, this, &FtmwConfigWidget::onSetDefaultPreset);

    // Repopulate preset combo on LoadoutManager state changes
    auto &lm = LoadoutManager::instance();
    connect(&lm, &LoadoutManager::ftmwPresetAdded,
            this, [this](auto, auto) { populatePresetCombo(); });
    connect(&lm, &LoadoutManager::ftmwPresetRemoved,
            this, [this](auto, auto) { populatePresetCombo(); });
    connect(&lm, &LoadoutManager::ftmwPresetChanged,
            this, [this](auto, auto) { populatePresetCombo(); });
    connect(&lm, &LoadoutManager::currentFtmwPresetChanged,
            this, [this](auto, auto) { populatePresetCombo(); });
    connect(&lm, &LoadoutManager::currentLoadoutChanged,
            this, [this](auto) { populatePresetCombo(); });

    populatePresetCombo();
}

void FtmwConfigWidget::initializeFromFtmwPreset(const FtmwPreset &preset)
{
    RfConfig rfc;
    preset.rfConfig.applyTo(rfc);
    p_rfWidget->setFromRfConfig(rfc);

    rfc.setChirpConfig(preset.chirpConfig);
    p_chirpWidget->setFromRfConfig(rfc);

    p_digiWidget->setFromConfig(preset.digitizer);
}

FtmwPreset FtmwConfigWidget::toFtmwPreset() const
{
    RfConfig rfc;
    p_rfWidget->toRfConfig(rfc);

    FtmwDigitizerConfig digiCfg(d_digiHwKey);
    p_digiWidget->toConfig(digiCfg);

    auto activeName = LoadoutManager::instance().currentLoadoutName();
    if (auto currentPreset = LoadoutManager::instance().currentFtmwPreset(activeName))
        digiCfg.d_fidChannel = currentPreset->digitizer.d_fidChannel;

    FtmwPreset result;
    result.rfConfig = RfConfigSnapshot::fromRfConfig(rfc);
    result.chirpConfig = p_chirpWidget->getChirps();
    result.digitizer = digiCfg;
    result.digiHwKey = d_digiHwKey;
    return result;
}

void FtmwConfigWidget::initializeFromExperiment(const FtmwConfig &cfg)
{
    p_rfWidget->setFromRfConfig(cfg.d_rfConfig);
    p_chirpWidget->setFromRfConfig(cfg.d_rfConfig);
    p_digiWidget->setFromConfig(cfg.scopeConfig());
}

void FtmwConfigWidget::resetToLoadout()
{
    auto activeName = LoadoutManager::instance().currentLoadoutName();
    if (auto preset = LoadoutManager::instance().currentFtmwPreset(activeName))
        initializeFromFtmwPreset(*preset);
}

void FtmwConfigWidget::updateChirpFromRf()
{
    RfConfig rfc;
    p_rfWidget->toRfConfig(rfc);
    p_chirpWidget->initialize(rfc);
    p_chirpWidget->updateChirpPlot();
}

void FtmwConfigWidget::clearDirty()
{
    const bool changed = d_dirty;
    d_dirty = false;
    if (changed)
        emit dirtyChanged(false);
    updatePresetBar();
}

void FtmwConfigWidget::markDirty()
{
    if (d_suppressDirty)
        return;
    const bool changed = !d_dirty;
    d_dirty = true;
    if (changed)
        emit dirtyChanged(true);
    updatePresetBar();
}

void FtmwConfigWidget::populatePresetCombo()
{
    const auto activeName = LoadoutManager::instance().currentLoadoutName();
    const auto presets = LoadoutManager::instance().ftmwPresetNames(activeName, false);
    const auto currentPresetName = LoadoutManager::instance().currentFtmwPresetName(activeName);

    p_ftmwPresetCombo->blockSignals(true);
    p_ftmwPresetCombo->clear();
    for (const auto &name : presets)
        p_ftmwPresetCombo->addItem(name);
    const int idx = p_ftmwPresetCombo->findText(currentPresetName);
    p_ftmwPresetCombo->setCurrentIndex(idx);
    p_ftmwPresetCombo->blockSignals(false);

    populateSourceCombos();
    updatePresetBar();
}

void FtmwConfigWidget::updatePresetBar()
{
    const auto activeName = LoadoutManager::instance().currentLoadoutName();
    const auto currentPresetName = LoadoutManager::instance().currentFtmwPresetName(activeName);
    const bool isReal = !currentPresetName.isEmpty()
        && currentPresetName != BC::Store::LM::lastUsedFtmwPresetName;
    const bool comboHasSelection = p_ftmwPresetCombo->currentIndex() >= 0;

    const QString comboName = p_ftmwPresetCombo->currentText();
    p_applyPresetButton->setEnabled(comboHasSelection
        && (comboName != currentPresetName || d_dirty));
    p_savePresetButton->setEnabled(isReal && d_dirty);
    p_saveAsPresetButton->setEnabled(true);
    p_renamePresetButton->setEnabled(isReal);
    p_deletePresetButton->setEnabled(isReal);
    p_setDefaultPresetButton->setEnabled(isReal);

    QString status;
    if (!isReal)
        status = "(unsaved)"_L1;
    if (d_dirty)
        status += status.isEmpty() ? "*"_L1 : " *"_L1;
    p_presetStatusLabel->setText(status);
    p_presetStatusLabel->setVisible(!status.isEmpty());
}

void FtmwConfigWidget::onApplyPreset()
{
    const QString name = p_ftmwPresetCombo->currentText();
    if (name.isEmpty())
        return;

    if (d_dirty) {
        const auto r = QMessageBox::question(
            this, "Load FTMW Preset"_L1,
            QString("Discard unsaved changes and load FTMW preset \"%1\"?").arg(name),
            QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
        if (r != QMessageBox::Yes)
            return;
    }

    const auto activeName = LoadoutManager::instance().currentLoadoutName();
    auto preset = LoadoutManager::instance().getFtmwPreset(activeName, name);
    if (!preset)
        return;

    initializeFromFtmwPreset(*preset);
    LoadoutManager::instance().setCurrentFtmwPresetName(activeName, name);

    RfConfig rfc;
    preset->rfConfig.applyTo(rfc);
    emit applyClocks(rfc.getClocks());

    clearDirty();
}

void FtmwConfigWidget::onSavePreset()
{
    const auto activeName = LoadoutManager::instance().currentLoadoutName();
    if (activeName.isEmpty())
        return;

    const auto currentPresetName = LoadoutManager::instance().currentFtmwPresetName(activeName);
    if (currentPresetName.isEmpty()
        || currentPresetName == BC::Store::LM::lastUsedFtmwPresetName)
        return;

    const auto preset = toFtmwPreset();
    LoadoutManager::instance().putFtmwPreset(activeName, currentPresetName, preset);
    LoadoutManager::instance().putFtmwPreset(
        activeName, BC::Store::LM::lastUsedFtmwPresetName, preset);
    clearDirty();
}

void FtmwConfigWidget::onSaveAsPreset()
{
    const auto activeName = LoadoutManager::instance().currentLoadoutName();
    if (activeName.isEmpty())
        return;

    bool ok;
    auto name = QInputDialog::getText(
        this, "Save FTMW Preset As"_L1, "Preset name:"_L1,
        QLineEdit::Normal, {}, &ok).trimmed();
    if (!ok || name.isEmpty())
        return;

    if (name == BC::Store::LM::lastUsedFtmwPresetName) {
        QMessageBox::warning(this, "Invalid Name"_L1, "That preset name is reserved."_L1);
        return;
    }

    if (LoadoutManager::instance().ftmwPresetExists(activeName, name)) {
        const auto r = QMessageBox::question(
            this, "Overwrite Preset"_L1,
            QString("Preset \"%1\" already exists. Overwrite?").arg(name),
            QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
        if (r != QMessageBox::Yes)
            return;
    }

    const auto preset = toFtmwPreset();
    LoadoutManager::instance().putFtmwPreset(activeName, name, preset);
    LoadoutManager::instance().putFtmwPreset(
        activeName, BC::Store::LM::lastUsedFtmwPresetName, preset);
    LoadoutManager::instance().setCurrentFtmwPresetName(activeName, name);
    clearDirty();
}

void FtmwConfigWidget::onRenamePreset()
{
    const auto activeName = LoadoutManager::instance().currentLoadoutName();
    if (activeName.isEmpty())
        return;

    const auto currentPresetName = LoadoutManager::instance().currentFtmwPresetName(activeName);
    if (currentPresetName.isEmpty()
        || currentPresetName == BC::Store::LM::lastUsedFtmwPresetName)
        return;

    bool ok;
    auto newName = QInputDialog::getText(
        this, "Rename FTMW Preset"_L1, "New name:"_L1,
        QLineEdit::Normal, currentPresetName, &ok).trimmed();
    if (!ok || newName.isEmpty() || newName == currentPresetName)
        return;

    if (newName == BC::Store::LM::lastUsedFtmwPresetName) {
        QMessageBox::warning(this, "Invalid Name"_L1, "That preset name is reserved."_L1);
        return;
    }

    if (LoadoutManager::instance().ftmwPresetExists(activeName, newName)) {
        QMessageBox::warning(this, "Name Exists"_L1,
            QString("A preset named \"%1\" already exists.").arg(newName));
        return;
    }

    LoadoutManager::instance().renameFtmwPreset(activeName, currentPresetName, newName);
}

void FtmwConfigWidget::onDeletePreset()
{
    const auto activeName = LoadoutManager::instance().currentLoadoutName();
    if (activeName.isEmpty())
        return;

    const auto currentPresetName = LoadoutManager::instance().currentFtmwPresetName(activeName);
    if (currentPresetName.isEmpty()
        || currentPresetName == BC::Store::LM::lastUsedFtmwPresetName)
        return;

    const auto r = QMessageBox::question(
        this, "Delete FTMW Preset"_L1,
        QString("Delete FTMW preset \"%1\"?").arg(currentPresetName),
        QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
    if (r != QMessageBox::Yes)
        return;

    LoadoutManager::instance().removeFtmwPreset(activeName, currentPresetName);
}

void FtmwConfigWidget::onSetDefaultPreset()
{
    const auto activeName = LoadoutManager::instance().currentLoadoutName();
    if (activeName.isEmpty())
        return;

    const auto currentPresetName = LoadoutManager::instance().currentFtmwPresetName(activeName);
    if (currentPresetName.isEmpty()
        || currentPresetName == BC::Store::LM::lastUsedFtmwPresetName)
        return;

    LoadoutManager::instance().setDefaultFtmwPresetName(activeName, currentPresetName);
}

void FtmwConfigWidget::setTabError(int tabIndex, bool hasError)
{
    if (hasError) {
        p_tabWidget->tabBar()->setTabTextColor(
            tabIndex, ThemeColors::getThemeAwareColor(ThemeColors::StatusError, this));
        p_tabWidget->setTabIcon(
            tabIndex, ThemeColors::createThemedIcon(
                ":/icons/exclamation-triangle.svg", ThemeColors::StatusError, this));
    } else {
        p_tabWidget->tabBar()->setTabTextColor(tabIndex, QColor());
        p_tabWidget->setTabIcon(tabIndex, QIcon());
    }
}

void FtmwConfigWidget::clearTabErrors()
{
    for (int i = 0; i < p_tabWidget->count(); ++i)
        setTabError(i, false);
}

void FtmwConfigWidget::populateSourceCombos()
{
    const QString sentinel = u"(no source — keep current)"_s;
    const auto activeName = LoadoutManager::instance().currentLoadoutName();
    const auto currentPresetName = LoadoutManager::instance().currentFtmwPresetName(activeName);

    auto presetNames = LoadoutManager::instance().ftmwPresetNames(activeName, false);
    presetNames.removeAll(currentPresetName);

    for (auto *combo : {p_rfSourceCombo, p_chirpSourceCombo, p_digiSourceCombo}) {
        combo->blockSignals(true);
        combo->clear();
        combo->addItem(sentinel);
        for (const auto &name : presetNames)
            combo->addItem(name, name);
        combo->setCurrentIndex(0);
        combo->blockSignals(false);
    }
}

void FtmwConfigWidget::onRfSourceChanged(int index)
{
    if (index <= 0)
        return;

    const QString sourceName = p_rfSourceCombo->currentData().toString();
    p_rfSourceCombo->blockSignals(true);
    p_rfSourceCombo->setCurrentIndex(0);
    p_rfSourceCombo->blockSignals(false);

    const auto activeName = LoadoutManager::instance().currentLoadoutName();
    auto sourcePreset = LoadoutManager::instance().getFtmwPreset(activeName, sourceName);
    if (!sourcePreset)
        return;

    RfConfig rfc;
    sourcePreset->rfConfig.applyTo(rfc);
    p_rfWidget->setFromRfConfig(rfc);
    markDirty();
}

void FtmwConfigWidget::onChirpSourceChanged(int index)
{
    if (index <= 0)
        return;

    const QString sourceName = p_chirpSourceCombo->currentData().toString();
    p_chirpSourceCombo->blockSignals(true);
    p_chirpSourceCombo->setCurrentIndex(0);
    p_chirpSourceCombo->blockSignals(false);

    const auto activeName = LoadoutManager::instance().currentLoadoutName();
    auto sourcePreset = LoadoutManager::instance().getFtmwPreset(activeName, sourceName);
    if (!sourcePreset)
        return;

    RfConfig rfc;
    sourcePreset->rfConfig.applyTo(rfc);
    rfc.setChirpConfig(sourcePreset->chirpConfig);
    p_chirpWidget->setFromRfConfig(rfc);
    markDirty();
}

void FtmwConfigWidget::onDigiSourceChanged(int index)
{
    if (index <= 0)
        return;

    const QString sourceName = p_digiSourceCombo->currentData().toString();
    p_digiSourceCombo->blockSignals(true);
    p_digiSourceCombo->setCurrentIndex(0);
    p_digiSourceCombo->blockSignals(false);

    const auto activeName = LoadoutManager::instance().currentLoadoutName();
    auto sourcePreset = LoadoutManager::instance().getFtmwPreset(activeName, sourceName);
    if (!sourcePreset)
        return;

    p_digiWidget->setFromConfig(sourcePreset->digitizer);
    markDirty();
}
